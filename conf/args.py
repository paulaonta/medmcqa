
from dataclasses import dataclass


@dataclass
class Arguments:
    train_csv:str
    test_csv:str 
    dev_csv:str
    incorrect_ans:int
    seed:int = 42
    batch_size:int = 4
    max_len:int = 192
    checkpoint_batch_size:int = 32
    print_freq:int = 100
    pretrained_model_name:str = "bert-base-uncased"#"google/bert_uncased_L-4_H-512_A-8" #bert-base-uncased"
    learning_rate:float = 2e-4
    hidden_dropout_prob:float =0.4
    hidden_size:int=768#512#768
    num_epochs:int = 5
    num_choices:int = 5
    device:str='cuda'
    gpu='0,1'
    use_context:bool=True
