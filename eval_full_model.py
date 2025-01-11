import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
from model.transformer import Transformer
from model.clip import CLIP

from eval_config import config

from eval_util import get_data_package, converter, tensor2str, \
     get_alphabet, convert_char,\
          get_radical_alphabet, load_vit_encoder_weights



alphabet = get_alphabet()
radical_alphabet = get_radical_alphabet()
print('alphabet', alphabet)

model = Transformer(config)

model = load_vit_encoder_weights(model, config['encoder_checkpoint_path']).cuda()
model = nn.DataParallel(model)

if config['resume'].strip() != '':
    checkpoint = torch.load(config['resume'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print('loading！！！')

test_loader = get_data_package()

# loading the pre-trained CCR-CLIP model
clip_model = CLIP(embed_dim=2048, context_length=30, vocab_size=len(radical_alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
clip_model = nn.DataParallel(clip_model)
clip_model.load_state_dict(torch.load(config['radical_model']), strict=False)
char_file = open(config['alpha_path'], 'r', encoding="utf-8").read()
chars = list(char_file)
tmp_text = convert_char(chars)
text_features = []
iters = len(chars) // 100
text_features.append(torch.zeros([1, 2048]).cuda())
with torch.no_grad():
    for i in range(iters+1):
        s = i * 100
        e = (i + 1) * 100
        if e > len(chars):
            e = len(chars)
        text_features_tmp = clip_model.module.encode_text(tmp_text[s:e])
        text_features.append(text_features_tmp)
    text_features.append(torch.ones([1, 2048]).cuda())
    text_features = torch.cat(text_features, dim=0).detach()


@torch.no_grad()
def test():

    torch.cuda.empty_cache()
    result_file = open(f"./{config['eval_dir']}/result_file_{config['eval_dataset']}.txt", 'w+', encoding='utf-8')

    print("Start Eval!")
    model.eval()
    dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    # test_loader_len = 10

    print('test:', test_loader_len)

    correct = 0
    total = 0

    for iteration in range(test_loader_len):
        data = dataloader.next()
        image, label, _ = data
        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

        length, text_input, text_gt, string_label = converter(label)
        max_length = max(length)
        batch = image.shape[0]
        pred = torch.zeros(batch,1).long().cuda()
        image_features = None
        prob = torch.zeros(batch, max_length).float()

        for i in range(max_length):
            length_tmp = torch.zeros(batch).long().cuda() + i + 1
            result = model(image, length_tmp, pred, conv_feature=image_features, test=True)

            prediction = result['pred'][:, -1:, :].squeeze()
            prediction = prediction / prediction.norm(dim=1, keepdim=True)
            prediction = prediction @ text_features.t()
            now_pred = torch.max(torch.softmax(prediction,1), 1)[1]
            prob[:,i] = torch.max(torch.softmax(prediction,1), 1)[0]
            pred = torch.cat((pred, now_pred.view(-1,1)), 1)
            image_features = result['conv']

        text_gt_list = []
        start = 0
        for i in length:
            text_gt_list.append(text_gt[start: start + i])
            start += i

        text_pred_list = []
        text_prob_list = []
        for i in range(batch):
            now_pred = []
            for j in range(max_length):
                if pred[i][j] != len(alphabet) - 1:
                    now_pred.append(pred[i][j])
                else:
                    break
            text_pred_list.append(torch.Tensor(now_pred)[1:].long().cuda())

            overall_prob = 1.0
            for j in range(len(now_pred)):
                overall_prob *= prob[i][j]
            text_prob_list.append(overall_prob)

        start = 0
        for i in range(batch):
            state = False
            pred = tensor2str(text_pred_list[i])
            gt = tensor2str(text_gt_list[i])

            if pred == gt:
                correct += 1
                state = True

            start += i
            total += 1
            result_file.write(
                '{} | {} | {} | {} | {} \n'.format(total, pred, gt, state, text_prob_list[i]))
        if (iteration + 1) % (test_loader_len // 10) == 0:
            print('{} | {} | {} | {} | {} | {} '.format(total, pred, gt, state, text_prob_list[i], correct / total))


    print("ACC : {}".format(correct/total))
    result_file.write("ACC : {}".format(correct/total))
    result_file.close()


if __name__ == '__main__':
    print('-------------')
    test()

    