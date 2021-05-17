import os

from transformers import BertTokenizer, TrainingArguments, LineByLineTextDataset
from transformers import Trainer, DataCollatorForLanguageModeling

from .nezha.model import NeZhaForMaskedLM
from .utils import seed_everyone

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_path():
    model_path = '../user_data/model_data/nezha-cn-base'
    data_path = '../user_data/duality_pair_pretrain_no_nsp_25w.tsv'
    output_path = '../user_data/tmp_data/pretrain_output/nezha_base_output'
    return model_path, data_path, output_path


def main():
    model_path, data_path, output_path = set_path()
    bort_tokenizer = BertTokenizer.from_pretrained(model_path)
    seed_everyone(20210409)
    dataset = LineByLineTextDataset(
        tokenizer=bort_tokenizer,
        file_path=data_path,
        block_size=42,
    )
    model = NeZhaForMaskedLM.from_pretrained(model_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=bort_tokenizer, mlm=True, mlm_probability=0.15
    )

    logging_path = os.path.join(output_path, 'log')
    model_save_path = os.path.join(output_path, 'best_model_ckpt')
    tokenizer_and_configs = os.path.join(output_path, 'tokenizer_and_configs')
    check_path(model_save_path)
    check_path(logging_path)
    check_path(tokenizer_and_configs)

    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        num_train_epochs=60,  # 60
        learning_rate=6e-5,
        fp16_backend='auto',
        per_device_train_batch_size=128,  # 64
        save_steps=1000,  # 1000
        logging_steps=1000,
        save_total_limit=10,  # 10
        run_name='80',
        logging_dir=logging_path,
        logging_first_step=True,
        dataloader_num_workers=4,
        disable_tqdm=False,
        seed=20200409
    )

    nezha_bert_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    nezha_bert_trainer.train()
    nezha_bert_trainer.save_model(model_save_path)
    bort_tokenizer.save_pretrained(tokenizer_and_configs)


if __name__ == '__main__':
    main()
