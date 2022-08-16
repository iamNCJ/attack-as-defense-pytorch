import numpy as np
import foolbox
import os
import argparse

from utils import attack_for_input, get_single_detector_train_data


def generate_training_data(model, data_loader):
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1))

    training_data_folder = 'cache/detector/'
    if not os.path.exists(training_data_folder):
        os.makedirs(training_data_folder)

    # just generate adversarial examples by untargeted attacks
    criterion = foolbox.criteria.Misclassification()
    adv_types_list = ['fgsm', 'bim', 'dba', 'df']

    for adv_type in adv_types_list:
        counter = 1
        if adv_type == 'fgsm':
            attack = foolbox.attacks.FGSM()
        elif adv_type == 'cw':
            attack = foolbox.attacks.L2CarliniWagnerAttack()
        elif adv_type == 'bim':
            attack = foolbox.attacks.LinfBasicIterativeAttack()
        elif adv_type == 'df':
            attack = foolbox.attacks.LinfDeepFoolAttack()
        elif adv_type == 'dba':
            attack = foolbox.attacks.BoundaryAttack()
        else:
            raise Exception('Unknown attacks method: ', str(adv_type))

        adversarial_result = []
        for img, label in data_loader:
            label = np.argmax(label)
            y_hat = fmodel(img)
            if y_hat != label:
                continue

            if adv_type == 'fgsm':
                adversarial = attack(fmodel, img, label, epsilon=0.03)
            elif adv_type == 'bim':
                adversarial = attack(fmodel, img, label, binary_search=False, epsilon=0.03, stepsize=0.003, steps=10)
            else:
                adversarial = attack(fmodel, img, label)

            if adversarial is not None:
                adv_label = np.argmax(fmodel(adversarial))
                if adv_label != label:
                    adversarial_result.append(adversarial)
                    counter += 1
                print('\r attacks success:', counter, end="")
                if counter == 250:
                    break

        print('\n%s attacks finished.' % adv_type)
        file_name = '%s_%s_adv_examples.npy' % (dataset, adv_type)
        np.save(training_data_folder + '/' + file_name, np.array(adversarial_result))


def generate_attack_cost(kmodel, dataset, attack_method):
    """ 
    Generate attacks costs for benign and adversarial examples.
    """

    x_input, y_input, _, _ = get_data(dataset)  # the input data min-max -> 0.0 1.0

    # except wrong label data
    preds_test = kmodel.predict_classes(x_input)
    inds_correct = np.where(preds_test == y_input.argmax(axis=1))[0]
    x_input = x_input[inds_correct]
    x_input = x_input[0: 1000]
    save_path = '../results/detector/' + dataset + '_benign'
    attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)

    for adv_type in ['fgsm', 'bim', 'jsma', 'df']:
        training_data_folder = '../results/detector/'
        file_name = '%s_%s_adv_examples.npy' % (dataset, adv_type)
        x_input = np.load(training_data_folder + file_name)
        save_path = '../results/detector/' + attack_method + '/' + dataset + '_' + adv_type
        attack_for_input(kmodel, x_input, y_input, attack_method, dataset=dataset, save_path=save_path)


def main(args):
    dataset = args.dataset
    attack_method = args.attack

    # * load model
    kmodel = load_model('../data/model_%s.h5' % dataset)

    if args.init:
        print('generate training data...')
        # ! Step 1. Generate training adv
        # * use this method to generate 1000 adversarial examples from training set as training set
        generate_training_data(kmodel, dataset)

        # ! Step 2. Get the attacks costs of training adv
        # * use this method to obtain atatck costs on training data
        generate_attack_cost(kmodel, dataset, attack_method)

    # ! Step 3. Train and pred
    root_path = '../results/'
    train_benign_data_path = 'detector/' + args.attack + '/' + args.dataset + '_benign'
    adv_types_list = ['fgsm', 'bim', 'jsma', 'df']
    test_types_list = ['benign', 'fgsm', 'bim-a', 'cw', 'jsma']
    train_adv_data_paths = []
    for adv_type in adv_types_list:
        train_adv_data_paths.append('detector/' + args.attack + '/' + args.dataset + '_' + adv_type)

    # * train k-nearest neighbors detector
    from sklearn import neighbors
    N = 100
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=N)
    print('knn based detector, k value is:', N)

    data_train, target_train = get_single_detector_train_data(root_path, train_benign_data_path,
                                                              train_adv_data_paths)
    knn_model.fit(data_train, target_train)
    print('training acc:', knn_model.score(data_train, target_train))

    # * test k-nearest neighbors detector
    for test_type in test_types_list:
        test_path = args.dataset + '_attack_iter_stats/' + args.attack + '_attack_' + test_type
        test_path = os.path.join(root_path, test_path)
        with open(test_path) as f:
            lines = f.read().splitlines()
        test_lines_list = []
        for line in lines:
            try:
                test_lines_list.append(int(line))
            except:
                raise Exception('Invalid data type in test data:', line)
        assert len(test_lines_list) == 1000

        test_lines_list = np.expand_dims(test_lines_list, axis=1)
        result = knn_model.predict(test_lines_list)
        acc = sum(result) / len(result)
        if test_type is 'benign':
            print('For knn based detector, detect acc on %s samples is %.4f' % (test_type, 1 - acc))
        else:
            print('For knn based detector, detect acc on %s samples is %.4f' % (test_type, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help="Dataset to use; either 'mnist', 'cifar'",
        required=False, type=str, default='mnist',
    )
    parser.add_argument(
        '--init',
        help="If this is the first time to run this script, \
            need to generate the training data and attacks costs.",
        action='store_true',
    )
    parser.add_argument(
        '-a', '--attacks',
        help="Attack to use; recommanded to use JSMA, BIM or BIM2.",
        required=True, type=str,
    )
    args = parser.parse_args()
    assert args.attack in ['JSMA', 'BIM', 'BIM2'], "Attack parameter error"

    args = parser.parse_args()
    main(args)
