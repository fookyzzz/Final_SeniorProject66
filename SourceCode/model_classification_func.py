'''
Classification of normal and abnormal symptoms of the gastrointestinal tract
from wireless capsule endoscopy with convolutional neural networks and transformer.

Function for Modeling
'''

import torch
import numpy as np
import time
import copy
from torch.autograd import Variable

from graph_func import show_databatch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Resuming Trainning Model Function


def train_model(model, criterion, optimizer, scheduler,
                train_data, val_data, train_dl, val_dl,
                device, select_best_value = "accuracy", use_gpu=False, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_specificity = 0.0
    best_f1_score = 0.0

    result_value = {
        "train_recall" : [],
        "train_precision" : [],
        "train_specificity" : [],
        "train_f1_score" : [],
        "train_tn" : [],
        "train_fp" : [],
        "train_fn" : [],
        "train_tp" : [],
        "train_cm" : [],
        "train_loss" : [],
        "train_acc" : [],
        "val_recall" : [],
        "val_precision" : [],
        "val_specificity" : [],
        "val_f1_score" : [],
        "val_tn" : [],
        "val_fp" : [],
        "val_fn" : [],
        "val_tp" : [],
        "val_cm" : [],
        "val_loss" : [],
        "val_acc" : []
    }

    num_epochs_best_state = 0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_dl)
    val_batches = len(val_dl)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        train_y_true = []
        train_y_pred = []
        val_y_true = []
        val_y_pred = []
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        model.to(device)
        
        for i, data in enumerate(train_dl):
            print("\rTraining batch {}/{} ({:.2f}%)".format(i, train_batches, (i * (100 / train_batches))), end='', flush=True)

            if i >= train_batches:
                break
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)

            for idx, y_true in enumerate(labels.data):
                train_y_true.append(y_true.cpu().numpy())
            for idx, y_pred in enumerate(preds):
                train_y_pred.append(y_pred.cpu().numpy())
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)

        train_recall = recall_score(train_y_true, train_y_pred)
        train_precision = precision_score(train_y_true, train_y_pred)
        train_cm = confusion_matrix(train_y_true, train_y_pred)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_specificity = train_tn / (train_tn + train_fp)
        train_f1_score = f1_score(train_y_true, train_y_pred)
        train_acc = accuracy_score(train_y_true, train_y_pred)

        result_value["train_recall"].append(train_recall)
        result_value["train_precision"].append(train_precision)
        result_value["train_specificity"].append(train_specificity)
        result_value["train_f1_score"].append(train_f1_score)

        result_value["train_tn"].append(train_tn)
        result_value["train_fp"].append(train_fp)
        result_value["train_fn"].append(train_fn)
        result_value["train_tp"].append(train_tp)
        result_value["train_cm"].append(train_cm)

        result_value["train_acc"].append(train_acc)
        
        model.train(False)
        model.eval()
    
        for i, data in enumerate(val_dl):
            print("\rValidation batch {}/{} ({:.2f}%)".format(i, val_batches, (i * (100 / val_batches))), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                # with torch.no_grad():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                # with torch.no_grad():
                    inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            for idx, y_true in enumerate(labels.data):
                val_y_true.append(y_true.cpu().numpy())
            for idx, y_pred in enumerate(preds):
                val_y_pred.append(y_pred.cpu().numpy())
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(val_data)
        avg_acc_val = acc_val / len(val_data)

        val_recall = recall_score(val_y_true, val_y_pred)
        val_precision = precision_score(val_y_true, val_y_pred)
        val_cm = confusion_matrix(val_y_true, val_y_pred)
        val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
        val_specificity = val_tn / (val_tn + val_fp)
        val_f1_score = f1_score(val_y_true, val_y_pred)
        val_acc = accuracy_score(val_y_true, val_y_pred)

        result_value["val_recall"].append(val_recall)
        result_value["val_precision"].append(val_precision)
        result_value["val_specificity"].append(val_specificity)
        result_value["val_f1_score"].append(val_f1_score)

        result_value["val_tn"].append(val_tn)
        result_value["val_fp"].append(val_fp)
        result_value["val_fn"].append(val_fn)
        result_value["val_tp"].append(val_tp)
        result_value["val_cm"].append(val_cm)

        result_value["val_acc"].append(val_acc)
        
        print()
        print("Epoch {} result: ".format(epoch + 1))
        print()
        print("Sensitivity (Recall) (train): {:.4f}".format(train_recall))
        print("Precision (train): {:.4f}".format(train_precision))
        print("Specificity (train): {:.4f}".format(train_specificity))
        print("F1 Score (train): {:.4f}".format(train_f1_score))
        print("Avg Loss (train): {:.4f}".format(avg_loss))
        print("ACCURACY (train): {:.4f}".format(train_acc))
        print()
        print("Sensitivity (Recall) (val): {:.4f}".format(val_recall))
        print("Precision (val): {:.4f}".format(val_precision))
        print("Specificity (val): {:.4f}".format(val_specificity))
        print("F1 Score (val): {:.4f}".format(val_f1_score))
        print("Avg Loss (val): {:.4f}".format(avg_loss_val))
        print("ACCURACY (val): {:.4f}".format(val_acc))
        print('-' * 10)
        print()

        result_value["train_loss"].append(avg_loss.cpu())
        result_value["val_loss"].append(avg_loss_val.cpu())
        
        slt_best_value = select_best_value.lower()

        if slt_best_value == "accuracy" and avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "recall" and val_recall > best_recall:
            best_recall = val_recall
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "precision" and val_precision > best_precision:
            best_precision = val_precision
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "specificity" and val_specificity > best_specificity:
            best_specificity = val_specificity
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch
        
        elif slt_best_value == "f1-score" and val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, result_value, num_epochs_best_state


# Resuming Trainning Model Function


def resume_train_model(model, criterion, optimizer, scheduler,
                       train_data, val_data, train_dl, val_dl,
                       device, results, select_best_value = "accuracy", use_gpu=False, num_epochs=10):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_specificity = 0.0
    best_f1_score = 0.0

    result_value = results

    num_epochs_best_state = 0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(train_dl)
    val_batches = len(val_dl)
    
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 10)

        train_y_true = []
        train_y_pred = []
        val_y_true = []
        val_y_pred = []
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        model.to(device)
        
        for i, data in enumerate(train_dl):
            print("\rTraining batch {}/{} ({:.2f}%)".format(i, train_batches, (i * (100 / train_batches))), end='', flush=True)

            if i >= train_batches:
                break
                
            inputs, labels = data
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_train += loss.data
            acc_train += torch.sum(preds == labels.data)

            for idx, y_true in enumerate(labels.data):
                train_y_true.append(y_true.cpu().numpy())
            for idx, y_pred in enumerate(preds):
                train_y_pred.append(y_pred.cpu().numpy())
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        print()
        # * 2 as we only used half of the dataset
        avg_loss = loss_train / len(train_data)
        avg_acc = acc_train / len(train_data)

        train_recall = recall_score(train_y_true, train_y_pred)
        train_precision = precision_score(train_y_true, train_y_pred)
        train_cm = confusion_matrix(train_y_true, train_y_pred)
        train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
        train_specificity = train_tn / (train_tn + train_fp)
        train_f1_score = f1_score(train_y_true, train_y_pred)
        train_acc = accuracy_score(train_y_true, train_y_pred)

        result_value["train_recall"].append(train_recall)
        result_value["train_precision"].append(train_precision)
        result_value["train_specificity"].append(train_specificity)
        result_value["train_f1_score"].append(train_f1_score)

        result_value["train_tn"].append(train_tn)
        result_value["train_fp"].append(train_fp)
        result_value["train_fn"].append(train_fn)
        result_value["train_tp"].append(train_tp)
        result_value["train_cm"].append(train_cm)

        result_value["train_acc"].append(train_acc)
        # result_value["train_acc"].append((torch.from_numpy(train_acc)).cpu())
        
        model.train(False)
        model.eval()
    
        for i, data in enumerate(val_dl):
            print("\rValidation batch {}/{} ({:.2f}%)".format(i, val_batches, (i * (100 / val_batches))), end='', flush=True)
                
            inputs, labels = data
            
            if use_gpu:
                # with torch.no_grad():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                # with torch.no_grad():
                    inputs, labels = Variable(inputs), Variable(labels)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            loss_val += loss.data
            acc_val += torch.sum(preds == labels.data)

            for idx, y_true in enumerate(labels.data):
                val_y_true.append(y_true.cpu().numpy())
            for idx, y_pred in enumerate(preds):
                val_y_pred.append(y_pred.cpu().numpy())
            
            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
        
        avg_loss_val = loss_val / len(val_data)
        avg_acc_val = acc_val / len(val_data)

        val_recall = recall_score(val_y_true, val_y_pred)
        val_precision = precision_score(val_y_true, val_y_pred)
        val_cm = confusion_matrix(val_y_true, val_y_pred)
        val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
        val_specificity = val_tn / (val_tn + val_fp)
        val_f1_score = f1_score(val_y_true, val_y_pred)
        val_acc = accuracy_score(val_y_true, val_y_pred)

        result_value["val_recall"].append(val_recall)
        result_value["val_precision"].append(val_precision)
        result_value["val_specificity"].append(val_specificity)
        result_value["val_f1_score"].append(val_f1_score)

        result_value["val_tn"].append(val_tn)
        result_value["val_fp"].append(val_fp)
        result_value["val_fn"].append(val_fn)
        result_value["val_tp"].append(val_tp)
        result_value["val_cm"].append(val_cm)

        result_value["val_acc"].append(val_acc)
        
        print()
        print("Epoch {} result: ".format(epoch + 1))
        print()
        print("Sensitivity (Recall) (train): {:.4f}".format(train_recall))
        print("Precision (train): {:.4f}".format(train_precision))
        print("Specificity (train): {:.4f}".format(train_specificity))
        print("F1 Score (train): {:.4f}".format(train_f1_score))
        print("Avg Loss (train): {:.4f}".format(avg_loss))
        print("ACCURACY (train): {:.4f}".format(train_acc))
        print()
        print("Sensitivity (Recall) (val): {:.4f}".format(val_recall))
        print("Precision (val): {:.4f}".format(val_precision))
        print("Specificity (val): {:.4f}".format(val_specificity))
        print("F1 Score (val): {:.4f}".format(val_f1_score))
        print("Avg Loss (val): {:.4f}".format(avg_loss_val))
        print("ACCURACY (val): {:.4f}".format(val_acc))
        print('-' * 10)
        print()

        result_value["train_loss"].append(avg_loss.cpu())
        result_value["val_loss"].append(avg_loss_val.cpu())
        
        slt_best_value = select_best_value.lower()

        if slt_best_value == "accuracy" and avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "recall" and val_recall > best_recall:
            best_recall = val_recall
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "precision" and val_precision > best_precision:
            best_precision = val_precision
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch

        elif slt_best_value == "specificity" and val_specificity > best_specificity:
            best_specificity = val_specificity
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch
        
        elif slt_best_value == "f1-score" and val_f1_score > best_f1_score:
            best_f1_score = val_f1_score
            best_model_wts = copy.deepcopy(model.state_dict())

            num_epochs_best_state = epoch
        
    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model, result_value, num_epochs_best_state


# Evaluate Model Function


def eval_model(model, criterion, class_names, test_dl, test_data, device, use_gpu=False):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0

    test_y_true = []
    test_y_pred = []

    result_value_test = {
        "test_recall" : [],
        "test_precision" : [],
        "test_specificity" : [],
        "test_f1_score" : [],
        "test_loss" : [],
        "test_acc" : [],
    }

    y_proba = []
    y_truth = []

    correctly_predicted_images = []
    incorrectly_predicted_images = []

    confusion_matrix_torch = torch.zeros(len(class_names), len(class_names))
    
    test_batches = len(test_dl)
    print("Evaluating model")
    print('-' * 10)
    # with torch.no_grad():
    for i, data in enumerate(test_dl):
        print("\rTest batch {}/{} ({:.2f}%)".format(i, test_batches, (i * (100 / test_batches))), end='', flush=True)

        model.train(False)
        model.eval()
        model.to(device)
        inputs, labels = data

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss.backward()

        loss_test += loss.data
        acc_test += torch.sum(preds == labels.data)

        for idx, y_true in enumerate(labels.data):
            test_y_true.append(y_true.cpu().numpy())
        for idx, y_pred in enumerate(preds):
            test_y_pred.append(y_pred.cpu().numpy())

        # TPR and FPR Value For ROC Curve
        for index, j in enumerate(outputs):
            y_proba.append(j[1])
            y_truth.append(labels[index])
            
        # Confusion Matrix Report
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix_torch[t.long(), p.long()] += 1

        # Separate correctly and incorrectly predicted images
        for j in range(len(labels)):
            if preds[j] == labels[j]:
                correctly_predicted_images.append((inputs[j], preds[j], labels[j]))
            else:
                incorrectly_predicted_images.append((inputs[j], preds[j], labels[j]))

        del inputs, labels, outputs, preds, loss
        torch.cuda.empty_cache()

    y_proba_out = np.array([float(y_proba[i]) for i in range(len(y_proba))])
    y_truth_out = np.array([float(y_truth[i]) for i in range(len(y_truth))])
        
    avg_loss = loss_test / len(test_data)
    avg_acc = acc_test / len(test_data)

    test_recall = recall_score(test_y_true, test_y_pred)
    test_precision = precision_score(test_y_true, test_y_pred)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(test_y_true, test_y_pred).ravel()
    test_specificity = test_tn / (test_tn + test_fp)
    test_f1_score = f1_score(test_y_true, test_y_pred)
    test_acc = accuracy_score(test_y_true, test_y_pred)

    result_value_test["test_recall"].append(test_recall)
    result_value_test["test_precision"].append(test_precision)
    result_value_test["test_specificity"].append(test_specificity)
    result_value_test["test_f1_score"].append(test_f1_score)
    result_value_test["test_loss"].append(avg_loss)
    result_value_test["test_acc"].append(test_acc)
    
    elapsed_time = time.time() - since
    print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Sensitivity (Recall) (test): {:.4f}".format(test_recall))
    print("Precision (test): {:.4f}".format(test_precision))
    print("Specificity (test): {:.4f}".format(test_specificity))
    print("F1 Score (test): {:.4f}".format(test_f1_score))
    print("Avg Loss (test): {:.4f}".format(avg_loss))
    print("ACCURACY (test): {:.4f}".format(test_acc))
    print('-' * 10)

    del model
    torch.cuda.empty_cache()

    return result_value_test, confusion_matrix_torch, y_proba_out, y_truth_out, correctly_predicted_images, incorrectly_predicted_images


# Visualize Image Classification


def visualize_model(model, test_dl, class_names, batch_size, use_gpu=False, num_images=6):
    was_training = model.training
    
    # Set model for evaluation
    model.train(False)
    model.eval() 
    
    images_so_far = 0
    with torch.no_grad():
        for i, data in enumerate(test_dl):
            inputs, labels = data
            size = inputs.size()[0]
            
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs.data, 1)
            predicted_labels = [preds[j] for j in range(inputs.size()[0])]
            
            print("Ground truth:")
            show_databatch(inputs.data.cpu(), labels.data.cpu(), class_names, batch_size)
            print("Prediction:")
            show_databatch(inputs.data.cpu(), predicted_labels, class_names, batch_size)
            
            del inputs, labels, outputs, preds, predicted_labels
            torch.cuda.empty_cache()
            
            images_so_far += size
            if images_so_far >= num_images:
                break
        
    model.train(mode=was_training) # Revert model back to original training state