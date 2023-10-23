import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from modeling_bart_ours import BartSimMCS
from modeling_pegasus_ours import PegasusSimMCS
from transformers import BartConfig


class SimMCS(nn.Module):
     
    def __init__(self, mname, pad_token_id, is_pegasus=False):
        super(SimMCS, self).__init__()
        config = BartConfig.from_pretrained(mname)
        if is_pegasus:
            self.model = PegasusSimMCS.from_pretrained(mname, cache_dir="./local_cache")
        else:
            self.model = BartSimMCS.from_pretrained(mname, cache_dir="./local_cache")

        # self.confidence_head = nn.Linear(config.d_model, 1, bias=True)
        self.pad_token_id = pad_token_id

    def forward(self, text_id, candidate_id, normalize=True, score_mode="base", length_penalty=1, require_gold=True, adding=0):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
   
        cand_mask[:, :, 0] = 1
   
        output = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=True
            )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]

        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        
        if normalize:
            if score_mode == "log":
                _output = F.log_softmax(output, dim=3)
            else:
                _output = F.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [bz, cand_num]
        
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        output['all_score'] = scores
        return output

    def scoring_mode(self):
        self.model.model.scoring_mode()

    def generation_mode(self):
        self.model.model.generation_mode()

    def generate(
        self,
        **model_kwargs,
    ):
        return self.model.generate(
            **model_kwargs)