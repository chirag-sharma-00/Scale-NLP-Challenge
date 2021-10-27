import numpy as np
import matplotlib.pyplot as plt
import re
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from main import load_file
from transformers import EncoderDecoderModel, BertTokenizer, BertConfig, EncoderDecoderConfig, get_linear_schedule_with_warmup

class ExpansionDataset(Dataset):
    def __init__(self, expansions_file, tokenizer):
        self.factors, self.expansions = load_file(expansions_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.factors)

    def __getitem__(self, idx):
        factor = self.factors[idx]
        match = re.search("[a-z](?!\(*[a-z]+)", factor)
        variable = factor[match.start(0):match.end(0)]
        data = re.sub("[a-z](?!\(*[a-z]+)", "x", factor)
        expansion = self.expansions[idx]
        label = re.sub("[a-z](?!\(*[a-z]+)", "x", expansion)
        return (data, label, variable)

    def collate_fn(self, batch):
        (data, labels, variables) = zip(*batch)
        data = self.tokenizer(list(data), text_pair=None,
                              return_tensors="pt", padding="max_length",
                              add_special_tokens=False,
                              return_token_type_ids=False)
        labels = self.tokenizer(list(labels), text_pair=None,
                              return_tensors="pt", padding="max_length",
                              add_special_tokens=False,
                              return_token_type_ids=False)
        return (data, labels, variables)


def create_model(tokens_file, num_layers):
    tokenizer = BertTokenizer(
        vocab_file="tokens.txt",
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=29
    )
    vocab_size = tokenizer.vocab_size
    max_input_len = tokenizer.model_max_length + 1
    encoder_config = BertConfig(vocab_size=vocab_size, 
                                num_hidden_layers=num_layers, 
                                hidden_size=256,
                                num_attention_heads=2,
                                max_position_embeddings=max_input_len,
                                sep_token_id=tokenizer.sep_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id)
    decoder_config = BertConfig(vocab_size=vocab_size, 
                                num_hidden_layers=num_layers, 
                                hidden_size=256,
                                num_attention_heads=2,
                                max_position_embeddings=max_input_len,
                                sep_token_id=tokenizer.sep_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                bos_token_id=tokenizer.bos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                decoder_start_token_id=tokenizer.bos_token_id)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = EncoderDecoderModel(config=config)
    return model, tokenizer

def train(train_file, val_file, model, tokenizer, lr, num_epochs, batch_size, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_data = ExpansionDataset(train_file, tokenizer)
    val_data = ExpansionDataset(val_file, tokenizer)
    num_samples = len(train_data)
    num_batches = num_samples // batch_size + 1
    num_train_steps = num_batches * num_epochs

    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)

    train_loss = []
    val_acc = []
    max_val_acc = -np.inf
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=val_data.collate_fn)
    for epoch in range(num_epochs):
        run_name = 'encoder_decoder_lr={}_epochs={}_batch_size={}_seed={}'.format(lr, epoch + 1, batch_size, seed)
        total_train_loss = 0
        print("Epoch {}:".format(epoch))
        print()
        model.train()
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            inputs, labels, variables = batch
            curr_batch_size = inputs.input_ids.shape[0]
            decoder_inputs = torch.cat([torch.LongTensor([tokenizer.bos_token_id]).expand(curr_batch_size, 1),
                                        labels.input_ids],
                                        dim=1)
            decoder_targets = torch.cat([labels.input_ids,
                                         torch.LongTensor([tokenizer.eos_token_id]).expand(curr_batch_size, 1)],
                                         dim=1)
            decoder_attention_mask = torch.cat([torch.LongTensor([1]).expand(curr_batch_size, 1),
                                                labels.attention_mask],
                                                dim=1)
            inputs = inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_targets = decoder_targets.to(device)
            decoder_attention_mask = decoder_attention_mask.to(device)
            pred = model(**inputs, decoder_input_ids=decoder_inputs, decoder_attention_mask=decoder_attention_mask)
            scores = pred.logits
            loss = criterion(scores.permute(0, 2, 1), decoder_targets)
            train_loss.append(loss.item())
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            gc.collect()
            del inputs
            del decoder_inputs
            del decoder_targets
            del decoder_attention_mask
            if (step % 100 == 0):
                print("Iteration {} / {}, loss = {}".format(step, num_batches, loss.item()))
                print()
        total_train_loss /= num_batches
        print("Epoch train loss =", total_train_loss)
        print()
        model.eval()
        batch_val_acc = []
        for step, batch in enumerate(val_dataloader):
            inputs, labels, variables = batch
            inputs = inputs.to(device)
            pred = tokenizer.batch_decode(model.generate(**inputs), skip_special_tokens=True)
            target = tokenizer.batch_decode(labels.input_ids, skip_special_tokens=True)
            correct = sum([pred[i] == target[i] for i in range(len(pred))])
            batch_val_acc.append(correct / len(pred))
            gc.collect()
            del inputs
        epoch_val_acc = np.mean(batch_val_acc)
        val_acc.append(epoch_val_acc)
        print("Epoch val acc =", epoch_val_acc)
        print()
        if epoch_val_acc > max_val_acc:
            max_val_acc = epoch_val_acc
            torch.save(model.state_dict(), run_name + ".pt")
        plt.figure()
        plt.title("Training loss")
        plt.plot(train_loss)
        plt.xlabel("Batch iteration")
        plt.savefig(run_name + "_trainloss.png")
        plt.figure()
        plt.title("Validation accuracy")
        plt.plot(val_acc)
        plt.xlabel("Epoch")
        plt.savefig(run_name + "_valacc.png")
    return None


def main():
    model, tokenizer = create_model(tokens_file="tokens.txt", num_layers=1)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters:", pytorch_total_params)
    print("Number of trainable parameters:", pytorch_trainable_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print()
    train(train_file="train.txt", val_file="val.txt",
          model=model, tokenizer=tokenizer, lr=5e-4, 
          num_epochs=3, batch_size=1024, seed=12321)
    

if __name__ == "__main__":
    main()

