import numpy as np
import torch
import torch.nn.functional as F
from horovod import torch as hvd
from transformers import BertTokenizer
from src.modeling.xbert import BertForMaskedLM, BertModel 
from transformers import ViTConfig , ViTModel , ViTFeatureExtractor
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from torch import nn


class PromuBaseModel(nn.Module):
    def __init__(self, config=None, input_format='RGB', image_enc_cfg=None, temp=0.07):
        super().__init__()
        
        self.temp = nn.Parameter(torch.ones([]) * temp)   

        self.bert_config = config

        visual_model_cls = ViTModel

        self.visual_encoder = visual_model_cls.from_pretrained("google/vit-base-patch16-224-in21k")
        self.text_encoder = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.image_processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        embed_dim = 256
        vision_width = 768

        text_width = self.bert_config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.cit_token_type = "cls"
        self.itm_head = nn.Linear(text_width, 2)     


    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

class PromuForPretrain(PromuBaseModel):
    def __init__(self, config, image_enc_cfg, input_format='RGB'):
        super(PromuForPretrain, self).__init__(config, input_format=input_format, image_enc_cfg=image_enc_cfg)

        self.prompter_image = Prompter(config, image_enc_cfg)
        self.prompter_obj = Prompter(config, image_enc_cfg)

        self.use_mask_prob = 0
        self.cir_head = nn.Sequential(
            nn.Linear(config.hidden_size,
                    config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 1000)
        )
        self.coe_head = nn.Sequential(
            nn.Linear(config.hidden_size,
                    config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, 1000)
        )
    def build_text_prompts(self, prompts,prompter):
        prompter.build_text_prompts(prompts)

    def get_pseudo_labels(self, batch ,prompter):
        return prompter.get_pseudo_labels(batch)

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']
        use_cir = 'cir_mask' in batch
        use_coe = 'coe_mask' in batch 
        itm_labels = batch['itm_labels']
        if use_cir and use_coe: 
            context_visual_inputs_image = batch['context_visual_inputs_image']
        device = visual_inputs.device
        b, c, h, w = visual_inputs.shape

        # forward image and text features
        # feats are normalized embeds
        if use_cir and use_coe and np.random.uniform() < self.use_mask_prob:
            image_embeds_total = self._forward_visual_embeds(torch.cat([visual_inputs, context_visual_inputs_image], dim=0))
            image_embeds, context_image_embeds = image_embeds_total[:b], image_embeds_total[b:]
        else:
            image_embeds = self._forward_visual_embeds(visual_inputs)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)
        # text embeddings and features
        text_embeds, text_feat = self._forward_text_feats(batch)

        gathered_image_feats = hvd.allgather(image_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        assert self.cit_token_type == 'cls', 'Support CLS tokens for CIT only, find {}.'.format(self.cit_token_type)
        sim_i2t = image_feat @ gathered_text_feats.t() / self.temp 
        sim_t2i = text_feat @ gathered_image_feats.t() / self.temp 

        sim_targets = torch.zeros_like(sim_i2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.diag(itm_labels) 
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_targets,dim=1).mean() 

        cit_loss = (loss_i2t+loss_t2i) / 2

        text_atts = batch['text_input_mask']

        itm_loss, itm_logits, itm_labels, encoder_outputs_pos = self.compute_itm(text_embeds=text_embeds, 
                                                                                 text_atts=text_atts, 
                                                                                 image_embeds=image_embeds, 
                                                                                 image_atts=image_atts, 
                                                                                 sim_i2t=sim_i2t.clone(), # for hard mining
                                                                                 sim_t2i=sim_t2i.clone(), # for hard mining
                                                                                 itm_labels = itm_labels,
                                                                                 return_encoder_out=True
                                                                                )
        cir_loss = cir_logits = cir_labels =  None
        coe_loss = coe_logits = coe_labels =  None

        if use_cir : 
            cir_labels, cir_ignore_masks= self.get_pseudo_labels(batch,self.prompter_image)
            cir_loss, cir_logits = self.compute_loss_with_encoder_out(encoder_outputs=encoder_outputs_pos, 
                                                            text_atts=text_atts, 
                                                            soft_labels=cir_labels, 
                                                            ignore_masks=cir_ignore_masks, 
                                                            patch_masks=batch['cir_mask'],
                                                            type = 'cir'
                                                        )
        if use_coe :
            coe_labels, coe_ignore_masks= self.get_pseudo_labels(batch,self.prompter_obj)
            coe_loss, coe_logits = self.compute_loss_with_encoder_out(encoder_outputs=encoder_outputs_pos, 
                                                            text_atts=text_atts, 
                                                            soft_labels=coe_labels, 
                                                            ignore_masks=coe_ignore_masks, 
                                                            patch_masks=batch['coe_mask'],
                                                            type = 'coe'
                                                        )



        return dict(
            cit_loss=cit_loss,
            itm_scores=itm_logits,  
            itm_loss=itm_loss,  
            itm_labels=itm_labels,   
            cir_loss=cir_loss,
            cir_logits=cir_logits,
            cir_labels=cir_labels,
            coe_loss=coe_loss,
            coe_logits=coe_logits,
            coe_labels=coe_labels
                )


    def _forward_visual_embeds(self, visual_inputs):
        b, c, h, w = visual_inputs.shape
        chunks = torch.chunk(visual_inputs, chunks=b, dim=0)
        visual_inputs = [chunk.squeeze(0).cpu() for chunk in chunks]
        inputs = self.image_processor(visual_inputs, return_tensors="pt")


        image_embeds = self.visual_encoder(**inputs.to('cuda'))

        return image_embeds.last_hidden_state

    def _forward_text_feats(self, batch):

        text_output = self.text_encoder.bert(batch['text_input_ids'], 
                                             attention_mask=batch['text_input_mask'],                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )

        text_embeds = text_output.last_hidden_state # b, Lt, fsz=768
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        return text_embeds, text_feat

    def compute_loss_with_encoder_out(self, encoder_outputs, text_atts, soft_labels, ignore_masks, patch_masks,type = 'cir'):
        txt_len = text_atts.shape[1]
        # adding one to ignore visual cls tokens
        visual_output = encoder_outputs.last_hidden_state[:, txt_len+1:]

        bsz, h, w = patch_masks.shape
        patch_masks_flatten_inverted = (1 - patch_masks.view(bsz, -1)).unsqueeze(-1)

        # mean embeds of masked visual regions
        num_masked_patches = torch.sum(patch_masks_flatten_inverted.squeeze(-1), dim=-1, keepdim=True)

        masked_visual_embeds = patch_masks_flatten_inverted * visual_output
        masked_visual_embeds = torch.sum(masked_visual_embeds, dim=1)
        masked_visual_embeds /= num_masked_patches

        # loss
        if type == 'cir':
            logits = self.cir_head(masked_visual_embeds)
        elif type == 'coe':
            logits = self.coe_head(masked_visual_embeds)
        cross_entropy = -torch.sum(F.log_softmax(logits, dim=1) * soft_labels, dim=1)
        cross_entropy[ignore_masks] = 0.

        loss = torch.sum(cross_entropy) / (bsz - torch.sum(ignore_masks))

        return loss, logits 

    def compute_itm(self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i,itm_labels, return_encoder_out=False):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )
        vl_embeddings = encoder_outputs_pos.last_hidden_state[:,0,:]
        itm_logits = self.itm_head(vl_embeddings)            
        itm_labels = itm_labels.to(device)
        itm_loss = F.cross_entropy(itm_logits, itm_labels)     

        if return_encoder_out:
            return itm_loss, itm_logits, itm_labels, encoder_outputs_pos 
        else:
            return itm_loss, itm_logits, itm_labels, None
        
    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None, prompter_weights_path_image=None,prompter_weights_path_obj=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)
        if prompter_weights_path_image is not None:
            self.prompter_image.load_pretrained_weights_without_prompts(prompter_weights_path_image)
        if prompter_weights_path_obj is not None:
            self.prompter_obj.load_pretrained_weights_without_prompts(prompter_weights_path_obj)


class Prompter(PromuBaseModel):
    def __init__(self, config, image_enc_cfg, input_format='RGB',type="image"):
        super(Prompter, self).__init__(config, input_format=input_format, image_enc_cfg=image_enc_cfg)
        self.type = type
        # self.entity_num = 1000
        if type == "image":
            self.relation_num = config.num_entities
        else:
            self.entity_num = config.num_entities
            
        self.register_buffer("image_prompt_feat", torch.rand(1000, 256)) 
        self.register_buffer("relation_prompt_feat", torch.rand(1000, 256)) 

        self.prompt_initialized = False
        # if the prob for the most likely entity is < 0.2, we just ignore it
        self.ignore_threshold = 0.2


    def load_pretrained_weights_without_prompts(self, ckpt_path):
        LOGGER.info("Loading weights for teacher model.")
        loaded_state_dict = torch.load(ckpt_path, map_location='cpu')

        loaded_keys = loaded_state_dict.keys()
        model_keys = self.state_dict().keys()

        load_not_in_model = [k for k in loaded_keys if k not in model_keys]
        model_not_in_load = [k for k in model_keys if k not in loaded_keys]

        if hvd.rank() == 0:
            LOGGER.info("Keys in loaded but not in model:")
            LOGGER.info(f"In total {len(load_not_in_model)}, {sorted(load_not_in_model)}")
            LOGGER.info("Keys in model but not in loaded:")
            LOGGER.info(f"In total {len(model_not_in_load)}, {sorted(model_not_in_load)}")

        new_loaded_state_dict = dict()
        for k in loaded_state_dict:
            if not 'prompt_feat' in k:
                new_loaded_state_dict[k] = loaded_state_dict[k]

        loaded_state_dict = new_loaded_state_dict

        self.load_state_dict(loaded_state_dict, strict=False)

    def build_text_prompts(self, prompts):

        assert not self.prompt_initialized, "Repetitively building prompts?"

        if self.training:
            self.eval()

        relation_prompt_feat_all = []
        image_prompt_feat_all = []

        with torch.no_grad():
            # this configurable depending on the GPU memory limit
            step_size = 10000

            # ====== initializing image prompting ======
            b_image, _ = prompts['batch_enc_image_prompts'].input_ids.shape

            start = 0
            end = start + step_size

            while start < b_image:
                if self.type == "object":
                    image_prompt_output = self.text_encoder.bert(prompts['batch_enc_image_prompts'].input_ids[start:end].cuda(), 
                                                                attention_mask=prompts['batch_enc_image_prompts'].attention_mask[start:end].cuda(),                      
                                                                return_dict=True, 
                                                                mode='text'
                                                                )

                    image_prompt_embeds = image_prompt_output.last_hidden_state # b, Lt, fsz=768
                    image_prompt_feat = F.normalize(self.text_proj(image_prompt_embeds[:,0,:]),dim=-1)      
                    image_prompt_feat_all.append(image_prompt_feat)

                else:
                    relation_prompt_output = self.text_encoder.bert(prompts['batch_enc_relation_prompts'].input_ids[start:end].cuda(), 
                                                attention_mask=prompts['batch_enc_relation_prompts'].attention_mask[start:end].cuda(),                      
                                                return_dict=True, 
                                                mode='text'
                                                )
                    relation_prompt_embeds = relation_prompt_output.last_hidden_state # b, Lt, fsz=768
                    relation_prompt_feat = F.normalize(self.text_proj(relation_prompt_embeds[:,0,:]),dim=-1)                 

                # collecting
                    relation_prompt_feat_all.append(relation_prompt_feat)
                start += step_size
                end += step_size

            # average ensembling
            if self.type == "object":

                image_prompt_feat = torch.cat(image_prompt_feat_all, dim=0)
                image_num_templates = int(image_prompt_feat.shape[0] / 1000)

                image_prompt_feat = torch.stack(image_prompt_feat.chunk(image_num_templates), dim=1)
                image_prompt_feat = torch.mean(image_prompt_feat, dim=1)
                self.image_prompt_feat = image_prompt_feat
            else:
                relation_prompt_feat = torch.cat(relation_prompt_feat_all, dim=0)
                relation_num_templates = int(relation_prompt_feat.shape[0] / 1000)

                relation_prompt_feat = torch.stack(relation_prompt_feat.chunk(relation_num_templates), dim=1)
                relation_prompt_feat = torch.mean(relation_prompt_feat, dim=1)
                self.relation_prompt_feat = relation_prompt_feat
           

        self.prompt_initialized = True

    def _forward_visual_embeds(self, visual_inputs):
        b, c, h, w = visual_inputs.shape
        chunks = torch.chunk(visual_inputs, chunks=b, dim=0)
        visual_inputs = [chunk.squeeze(0).cpu() for chunk in chunks]
        inputs = self.image_processor(visual_inputs, return_tensors="pt")


        image_embeds = self.visual_encoder(**inputs.to('cuda')).last_hidden_state

        assert self.cit_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.cit_token_type)
        if self.cit_token_type == 'cls':
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
        else:
            raise NotImplementedError("cit_type_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.cit_token_type))
        
        return image_embeds, image_feat

    def _compute_soft_labels(self, sim_vp_masked):
        soft_labels = nn.Softmax(dim=1)(sim_vp_masked)
        ignore_masks = torch.max(sim_vp_masked, dim=1)[1] < self.ignore_threshold

        return soft_labels, ignore_masks

    def get_pseudo_labels(self, batch):
        if self.training:
            self.eval()

        with torch.no_grad():
            masked_visual_inputs_image = batch['context_visual_inputs_image']
            masked_visual_inputs_object = batch['crop_visual_inputs_image']
            if self.type == "image":
                _, masked_image_feat = self._forward_visual_embeds(masked_visual_inputs_image)
                sim_masked_image = masked_image_feat @ self.relation_prompt_feat.t() / self.temp 
                pseudo_labels_image, ignore_masks_image = self._compute_soft_labels(sim_masked_image)
                return pseudo_labels_image, ignore_masks_image

            else:
                _, masked_object_feat = self._forward_visual_embeds(masked_visual_inputs_object)
                sim_masked_object = masked_object_feat @ self.image_prompt_feat.t() / self.temp 
                pseudo_labels_object , ignore_masks_object = self._compute_soft_labels(sim_masked_object)
                return pseudo_labels_object ,ignore_masks_object

    def forward(self, batch):
        visual_inputs = batch['visual_inputs']
        itm_labels = batch ['itm_labels']
        device = visual_inputs.device
        b, c, h, w = visual_inputs.shape

        # forward image and text features
        # feats are normalized embeds
        image_embeds, image_feat, text_embeds, text_feat = self.forward_feats(batch)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # ========== (in-batch) ITC loss ==========
        gathered_image_feats = hvd.allgather(image_feat)
        gathered_text_feats = hvd.allgather(text_feat)

        assert self.cit_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.cit_token_type)

        sim_i2t = image_feat @ gathered_text_feats.t() / self.temp 
        sim_t2i = text_feat @ gathered_image_feats.t() / self.temp 
                             
        # [IMPORTANT] be very careful when initializing the GT sim_i2t 
        # allgather return the concatenated features in the order of local_rank()
        sim_targets = torch.zeros_like(sim_i2t)

        local_rank = hvd.local_rank()
        b_start, b_end = b * local_rank, b * (local_rank + 1)
        sim_targets[:, b_start: b_end] = torch.torch.diag(itm_labels) 

        sim_i2t_scores = F.log_softmax(sim_i2t, dim=1)
        sim_t2i_scores = F.log_softmax(sim_t2i, dim=1)

        loss_i2t = -torch.sum(sim_i2t_scores * sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(sim_t2i_scores * sim_targets,dim=1).mean() 

        cit_loss = (loss_i2t+loss_t2i) / 2

        return dict(
            cit_loss=cit_loss,
            cit_labels=torch.max(sim_targets, dim=1)[1],
            i2t_scores=sim_i2t_scores,
            t2i_scores=sim_t2i_scores
        )


    def forward_feats(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']

        b, c, h, w = visual_inputs.shape
        chunks = torch.chunk(visual_inputs, chunks=b, dim=0)
        visual_inputs = [chunk.squeeze(0).cpu() for chunk in chunks]
        inputs = self.image_processor(visual_inputs, return_tensors="pt")


        image_embeds = self.visual_encoder(**inputs.to('cuda')).last_hidden_state

        assert self.cit_token_type == 'cls', 'Expecting CLS token for ITC, found {}'.format(self.cit_token_type)
        if self.cit_token_type == 'cls':
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)  
        else:
            raise NotImplementedError("cit_type_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.cit_token_type))

        # text features
        text_output = self.text_encoder.bert(batch['text_input_ids'], 
                                             attention_mask=batch['text_input_mask'],                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )

        text_embeds = text_output.last_hidden_state # b, Lt, fsz=768

        if self.cit_token_type == 'cls':
            text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
        else:
            raise NotImplementedError("cit_token_type must be one of ['mean', 'cls', 'mil'], found {}".format(self.cit_token_type))

        return image_embeds, image_feat, text_embeds, text_feat


class PromuForRe(PromuBaseModel):
    def __init__(self, config, image_enc_cfg, input_format='RGB'):
        super(PromuForRe, self).__init__(config, image_enc_cfg=image_enc_cfg)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config, add_pooling_layer=False)      
        self.text_encoder.resize_token_embeddings(len(tokenizer)+4)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2,
                      config.hidden_size),
            nn.ReLU(True),
            nn.Linear(config.hidden_size, config.num_labels)
        )

    # def forward(self, image, text, targets, alpha=0, train=True):
    def forward(self, batch):
        visual_inputs = batch['visual_inputs']
        targets = batch['labels']

        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder(input_ids=batch['text_input_ids'],
                                        attention_mask=text_input_mask,
                                        return_dict=True,
                                        )
        text_embeds = text_output.last_hidden_state

        head_idx = batch['text_input_ids'].eq(self.head_start).nonzero()[:, 1].unsqueeze(1)
        tail_idx = batch['text_input_ids'].eq(self.tail_start).nonzero()[:, 1].unsqueeze(1)


        text_embeds = text_output.last_hidden_state

        # forward visual
        b, c, h, w = visual_inputs.shape
        chunks = torch.chunk(visual_inputs, chunks=b, dim=0)
        visual_inputs = [chunk.squeeze(0).cpu() for chunk in chunks]
        inputs = self.image_processor(visual_inputs, return_tensors="pt")


        image_embeds = self.visual_encoder(**inputs.to('cuda')).last_hidden_state
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)
        output = self.text_encoder(encoder_embeds=embedding_output,
                                attention_mask=attention_mask,
                                return_dict=True,
                                ).last_hidden_state
        head_hidden = output.gather(1, head_idx.unsqueeze(2).expand(-1, -1, output.size(-1)))
        tail_hidden = output.gather(1, tail_idx.unsqueeze(2).expand(-1, -1, output.size(-1)))
        output = torch.cat([head_hidden,tail_hidden],dim=-1)
        prediction = self.classifier(output[:,0,:])     
        if targets is not None:
            loss = F.cross_entropy(prediction, targets)                
        else: # evaluation mode
            loss = 0

        return dict(loss=loss,
                    logits=prediction
                    )
            

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder.bert(batch['text_input_ids'],
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                            )
        text_embeds = text_output.last_hidden_state

        # forward visual
        b, c, h, w = visual_inputs.shape
        chunks = torch.chunk(visual_inputs, chunks=b, dim=0)
        visual_inputs = [chunk.squeeze(0).cpu() for chunk in chunks]
        inputs = self.image_processor(visual_inputs, return_tensors="pt")


        image_embeds = self.visual_encoder(**inputs.to('cuda')).last_hidden_state
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = self.text_encoder.bert(encoder_embeds=embedding_output,
                                        attention_mask=attention_mask,
                                        return_dict=True,
                                        mode='fusion'
                                    )

        prediction = self.classifier(output.last_hidden_state[:,0,:])                

        return prediction

