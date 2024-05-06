'''
Classification of normal and abnormal symptoms of the gastrointestinal tract
from wireless capsule endoscopy with convolutional neural networks and transformer.

Function for Plot Graph
'''

import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Plot Image

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes, class_names, batch_size):
    out = torchvision.utils.make_grid(inputs)
    class_title = ""
    count = 0
    for x in classes:
        if count%8 == 0:
            class_title = class_title + "\n" + "[" + str(class_names[x]) + "] , "
        elif count == batch_size:
            class_title = class_title + "\n" + "[" + str(class_names[x]) + "]"
        else:
            class_title = class_title + "[" + str(class_names[x]) + "] , "
        count = count + 1
        
    imshow(out, title=class_title)

def show_databatch_normalize(inputs, classes, class_names, batch_size):

    original_inputs = inputs * torch.tensor(std).view(1, 3, 1, 1) + torch.tensor(mean).view(1, 3, 1, 1)

    out = torchvision.utils.make_grid(original_inputs)
    class_title = ""
    count = 0
    for x in classes:
        if count%8 == 0:
            class_title = class_title + "\n" + "[" + str(class_names[x]) + "] , "
        elif count == batch_size:
            class_title = class_title + "\n" + "[" + str(class_names[x]) + "]"
        else:
            class_title = class_title + "[" + str(class_names[x]) + "] , "
        count = count + 1
        
    imshow(out, title=class_title)

# Plot Accuracy and Loss

def plot_loss_curves(results, now, path_model):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    loss = results["train_loss"]
    test_loss = results["val_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]

    # Average Value
    avg_loss = sum(loss) / len(loss)
    avg_test_loss = sum(test_loss) / len(test_loss)

    avg_acc = sum(accuracy) / len(accuracy)
    avg_test_acc = sum(test_accuracy) / len(test_accuracy)

    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Train and Validation Loss\nAverage Train Loss : {:.4f}\nAverage Validation Loss : {:.4f}".format(avg_loss, avg_test_loss))
    plt.xlabel("Epochs")
    plt.ylim([0.000, 0.025])
    plt.grid(True)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Train and Validation Accuracy\nAverage Train Accuracy : {:.4f}\nAverage Validation Accuracy : {:.4f}".format(avg_acc, avg_test_acc))
    plt.xlabel("Epochs")
    plt.ylim([0.5, 1.05])
    plt.grid(True)
    plt.legend()

    dt_string = now.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{path_model}/Loss_and_Accuracy_{dt_string}.png')

# Scoring of training and validation graph plotting functions.
def plot_train_validate_score(results, now, path_model):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    train_recall = results["train_recall"]
    train_precision = results["train_precision"]
    train_specificity = results["train_specificity"]
    train_f1_score = results["train_f1_score"]
    val_recall = results["val_recall"]
    val_precision = results["val_precision"]
    val_specificity = results["val_specificity"]
    val_f1_score = results["val_f1_score"]

    epochs = range(1, len(results["train_loss"]) + 1)

    plt.figure(figsize=(15, 7))

    # Plot training score
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_recall, label="train_recall")
    plt.plot(epochs, train_precision, label="train_precision")
    plt.plot(epochs, train_specificity, label="train_specificity")
    plt.plot(epochs, train_f1_score, label="train_f1_score")
    plt.title("Training Score")
    plt.xlabel("Epochs")
    plt.ylim([0.000, 1.000])
    plt.grid(True)
    plt.legend()

    # Plot validation score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_recall, label="val_recall")
    plt.plot(epochs, val_precision, label="val_precision")
    plt.plot(epochs, val_specificity, label="val_specificity")
    plt.plot(epochs, val_f1_score, label="val_f1_score")
    plt.title("Validation Score")
    plt.xlabel("Epochs")
    plt.ylim([0.000, 1.000])
    plt.grid(True)
    plt.legend()

    dt_string = now.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{path_model}/Training_and_Validation_Score_{dt_string}.png')

# Test Predict and Plot Image

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    device: torch.device,
    image_size: Tuple[int, int] = (224, 224),
    transform: torchvision.transforms = None,
):
    """Predicts on a target image with a target model.

    Args:
        model (torch.nn.Module): A trained (or untrained) PyTorch model to predict on an image.
        class_names (List[str]): A list of target classes to map predictions to.
        image_path (str): Filepath to target image to predict on.
        image_size (Tuple[int, int], optional): Size to transform target image to. Defaults to (224, 224).
        transform (torchvision.transforms, optional): Transform to perform on image. Defaults to None which uses ImageNet normalization.
        device (torch.device, optional): Target device to perform prediction on. Defaults to device.
    """

    # Open image
    img = Image.open(image_path)

    # Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    ### Predict on image ###

    # Make sure the model is on the target device
    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

# Plot Confusion Martrix

def plot_confusion_matrix (cm: [], class_names, path_cm, cm_filename):

    new_cm = cm[[1, 0]][:, [1, 0]]

    cm_plot = sns.heatmap(new_cm,
                annot=True,
                fmt='g',
                xticklabels=class_names,
                yticklabels=class_names)
    fig = cm_plot.get_figure()
    plt.ylabel('ACTUAL',fontsize=13)
    plt.xlabel('PREDICTION',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()

    # dt_string = now.strftime("%Y%m%d_%H%M%S")

    fig.savefig(f'{path_cm}/{cm_filename}.png')

# Plot ROC and AUC Score

def plot_ROCAUC_curve(y_truth, y_proba, fig_size, now, path_model):
    
    '''
    Plots the Receiver Operating Characteristic Curve (ROC) and displays Area Under the Curve (AUC) score.
    
    Args:
        y_truth: ground truth for testing data output
        y_proba: class probabilties predicted from model
        fig_size: size of the output pyplot figure
    
    Returns: void
    '''
    
    fpr, tpr, threshold = roc_curve(y_truth, y_proba)
    auc_score = roc_auc_score(y_truth, y_proba)
    txt_box = "AUC Score: " + str(round(auc_score, 4))
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'--')
    plt.annotate(txt_box, xy=(0.65, 0.05), xycoords='axes fraction')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")

    dt_string = now.strftime("%Y%m%d_%H%M%S")

    plt.savefig(f'{path_model}/ROC_Curve_{dt_string}.png')