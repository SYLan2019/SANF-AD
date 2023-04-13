import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tqdm import tqdm
from model import load_model
import config as c
from utils import *
import matplotlib.pyplot as plt
import torch.nn.functional as F
import PIL
from os.path import join
import os
from copy import deepcopy


# localize = True
# upscale_mode = 'bilinear'
score_export_dir = join('./viz/scores/', c.modelname)
os.makedirs(score_export_dir, exist_ok=True)
# map_export_dir = join('./viz/maps/', c.modelname)
# os.makedirs(map_export_dir, exist_ok=True)

#scores classes = labels
def compare_histogram(scores, classes, bins1=100,bins2 =100,class_name=''):
    classes = deepcopy(classes)
    classes[classes > 0] = 1
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins=bins1, alpha=0.5, density=False, label='normal({})'.format(class_name), color='blue', edgecolor="black")
    plt.hist(scores_ano, bins=bins2, alpha=0.5, density=False, label='abnormal', color='orange', edgecolor="black")
    plt.xlabel(r'$-log(p(z)) or Anomaly Score$')
    plt.ylabel('Count ')
    plt.legend()
    plt.grid(axis='y')
    save_dir = join(score_export_dir,c.dataset)
    os.makedirs(save_dir,exist_ok=True)
    plt.savefig(join(save_dir,class_name + '_score_histogram.png'), bbox_inches='tight', pad_inches=0)


# def viz_roc(values, classes, class_names):
#     def export_roc(values, classes, export_name='all'):
#         # Compute ROC curve and ROC area for each class
#         classes = deepcopy(classes)
#         classes[classes > 0] = 1
#         fpr, tpr, _ = roc_curve(classes, values)
#         roc_auc = auc(fpr, tpr)
#
#         plt.clf()
#         lw = 2
#         plt.plot(fpr, tpr, color='darkorange',
#                  lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
#
#         plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver operating characteristic for class ' + c.class_name)
#         plt.legend(loc="lower right")
#         plt.axis('equal')
#         plt.xlim([0.0, 1.0])
#         plt.ylim([0.0, 1.0])
#         plt.savefig(join(score_export_dir, export_name + '.png'))
#
#     export_roc(values, classes)
#     for cl in range(1, classes.max() + 1):
#         filtered_indices = np.concatenate([np.where(classes == 0)[0], np.where(classes == cl)[0]])
#         classes_filtered = classes[filtered_indices]
#         values_filtered = values[filtered_indices]
#         export_roc(values_filtered, classes_filtered, export_name=class_names[filtered_indices[-1]])
#

# def viz_maps(maps, name, label):
#     img_path = img_paths[c.viz_sample_count]
#     image = PIL.Image.open(img_path).convert('RGB')
#     image = np.array(image)
#
#     map_to_viz = t2np(F.interpolate(maps[0][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
#         0, 0]
#
#     plt.clf()
#     plt.imshow(map_to_viz)
#     plt.axis('off')
#     plt.savefig(join(map_export_dir, name + '_map.jpg'), bbox_inches='tight', pad_inches=0)
#
#     if label > 0:
#         plt.clf()
#         plt.imshow(image)
#         plt.axis('off')
#         plt.savefig(join(map_export_dir, name + '_orig.jpg'), bbox_inches='tight', pad_inches=0)
#         plt.imshow(map_to_viz, cmap='viridis', alpha=0.3)
#         plt.savefig(join(map_export_dir, name + '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
#     return
# def viz_map_array(maps, labels, n_col=8, subsample=4, max_figures=-1):
#     plt.clf()
#     fig, subplots = plt.subplots(3, n_col)
#
#     fig_count = -1
#     col_count = -1
#     for i in range(len(maps)):
#         if i % subsample != 0:
#             continue
#
#         if labels[i] == 0:
#             continue
#
#         col_count = (col_count + 1) % n_col
#         if col_count == 0:
#             if fig_count >= 0:
#                 plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
#                 plt.close()
#             fig, subplots = plt.subplots(3, n_col, figsize=(22, 8))
#             fig_count += 1
#             if fig_count == max_figures:
#                 return
#
#         anomaly_description = img_paths[i].split('/')[-2]
#         image = PIL.Image.open(img_paths[i]).convert('RGB')
#         image = np.array(image)
#         map = t2np(F.interpolate(maps[i][None, None], size=image.shape[:2], mode=upscale_mode, align_corners=False))[
#             0, 0]
#         subplots[1][col_count].imshow(map)
#         subplots[1][col_count].axis('off')
#         subplots[0][col_count].imshow(image)
#         subplots[0][col_count].axis('off')
#         subplots[0][col_count].set_title(c.class_name + ":\n" + anomaly_description)
#         subplots[2][col_count].imshow(image)
#         subplots[2][col_count].axis('off')
#         subplots[2][col_count].imshow(map, cmap='viridis', alpha=0.3)
#     for i in range(col_count, n_col):
#         subplots[0][i].axis('off')
#         subplots[1][i].axis('off')
#         subplots[2][i].axis('off')
#     if col_count > 0:
#         plt.savefig(join(map_export_dir, str(fig_count) + '.jpg'), bbox_inches='tight', pad_inches=0)
#     return


def evaluate(model, test_loader):
    model.to(c.device)
    model.eval()
    print('\nCompute maps, loss and scores on test set:')
    anomaly_score = list()
    test_labels = list()
    c.viz_sample_count = 0
    # all_maps = list()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            # if not c.pre_extracted:
            #     inputs = fe(inputs)
            z = model(inputs)
            # z_concat = t2np(concat_maps(z))
            # t2np(torch.mean((z ** 2)/2, dim=1))
            nll_score = np.mean(t2np(z) ** 2 / 2, axis=(1))
            anomaly_score.append(nll_score)
            test_labels.append(t2np(labels))

            # if localize:
            #     z_grouped = list()
            #     likelihood_grouped = list()
            #     for i in range(len(z)):
            #         z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
            #         likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)) / c.n_feat)
            #     all_maps.extend(likelihood_grouped[0])
            #     for i_l, l in enumerate(t2np(labels)):
            #         # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
            #         c.viz_sample_count += 1

    anomaly_score = np.concatenate(anomaly_score)
    test_labels = np.concatenate(test_labels)

    compare_histogram(anomaly_score, test_labels,class_name=c.class_name)

    # class_names = [img_path.split('/')[-2] for img_path in img_paths]
    # viz_roc(anomaly_score, test_labels, class_names)

    test_labels = np.array([1 if l > 0 else 0 for l in test_labels])
    auc_score = roc_auc_score(test_labels, anomaly_score)
    print('AUC:', auc_score)
    # if localize:
    #     viz_map_array(all_maps, test_labels)

    return

# _, test_loader = get_loaders(c.dataset, c.class_name,batch_size=c.batch_size)
# mod = load_model(c.modelname)
# evaluate(mod, test_loader)
