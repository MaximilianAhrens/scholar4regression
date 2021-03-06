import os
import sys
from optparse import OptionParser

import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

import file_handling as fh
from scholar_tf import Scholar


def main():
    start_time = datetime.now()
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-a', dest='alpha', default=1.0,
                      help='Hyperparameter for logistic normal prior: default=%default')
    parser.add_option('-k', dest='n_topics', default=20,
                      help='Size of latent representation (~num topics): default=%default')
    parser.add_option('--batch-size', dest='batch_size', default=200,
                      help='Size of minibatches: default=%default')
    parser.add_option('-l', dest='learning_rate', default=0.002,
                      help='Initial learning rate: default=%default')
    parser.add_option('-m', dest='momentum', default=0.99,
                      help='beta1 for Adam: default=%default')
    parser.add_option('--epochs', dest='epochs', default=200,
                      help='Number of epochs: default=%default')
    parser.add_option('--train-prefix', type=str, default='train',
                      help='Prefix of train set: default=%default')
    parser.add_option('--test-prefix', dest='test_prefix', default=None,
                      help='Prefix of test set: default=%default')
    parser.add_option('--en-layers', dest='encoder_layers', default=1,
                      help='Number of encoder layers [0|1|2]: default=%default')
    parser.add_option('--emb-dim', dest='embedding_dim', default=300,
                      help='Dimension of input embeddings: default=%default')
    parser.add_option('--en-short', action="store_true", dest="encoder_shortcuts", default=False,
                      help='Use shortcut connections on encoder: default=%default')
    parser.add_option('--labels', dest='label_name', default=None,
                      help='Read labels from input_dir/[train|test]_prefix.label_name.csv: default=%default')
    parser.add_option('--topic-covars', dest='covar_names', default=None,
                      help='Read covars from files with these names (comma-separated): default=%default')
    parser.add_option('--label-emb-dim', dest='label_emb_dim', default=-1,
                      help="Class embedding dimension [-1 = identity; 0 = don't encode]: default=%default")
    parser.add_option('--covar-emb-dim', dest='covar_emb_dim', default=-1,
                      help="Covariate embedding dimension [-1 = identity; 0 = don't encode]: default=%default")
    parser.add_option('--min-covar-count', dest='min_covar_count', default=None,
                      help='Drop binary covariates that occur less than this in training: default=%default')
    parser.add_option('--interactions', action="store_true", dest="covar_interactions", default=False,
                      help='Use covariate interactions in model: default=%default')
    parser.add_option('--infer-covars', action="store_true", dest="infer_covars", default=False,
                      help='Infer categorical covariate values after fitting model (slow): default=%default')
    parser.add_option('--c-layers', dest='classifier_layers', default=1,
                      help='Number of layers in (generative) classifier [0|1|2]: default=%default')
    parser.add_option('--exclude-covars', action="store_true", dest="exclude_covars", default=False,
                      help='Exclude covariates from the classifier: default=%default')
    parser.add_option('-r', action="store_true", dest="regularize", default=False,
                      help='Apply adaptive regularization for sparsity in topics: default=%default')
    parser.add_option('-o', dest='output_dir', default='output',
                      help='Output directory: default=%default')
    parser.add_option('--w2v', dest='word2vec_file', default=None,
                      help='Use this word2vec .bin file to initialize and fix embeddings: default=%default')
    parser.add_option('--vocab-size', dest='vocab_size', default=None,
                      help='Filter the vocabulary keeping the most common n words: default=%default')
    parser.add_option('--update-bg', action="store_true", dest="update_bg", default=False,
                      help='Update background parameters: default=%default')
    parser.add_option('--no-bg', action="store_true", dest="no_bg", default=False,
                      help='Do not use background freq: default=%default')
    parser.add_option('--no-bn-anneal', action="store_true", dest="no_bn_anneal", default=False,
                      help='Do not anneal away from batchnorm: default=%default')
    parser.add_option('--samples', dest='test_samples', default=20,
                      help='Number of samples to use in computing test perplexity: default=%default')
    parser.add_option('--dev-folds', dest='dev_folds', default=0,
                      help='Number of dev folds: default=%default')
    parser.add_option('--dev-fold', dest='dev_fold', default=0,
                      help='Fold to use as dev (if dev_folds > 0): default=%default')
    parser.add_option('--opt', dest='optimizer', default='adam',
                      help='Optimization algorithm to use [adam|adagrad|sgd]: default=%default')
    parser.add_option('--threads', dest='threads', default=8,
                      help='Use this to limit the number of CPUs: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')
    # own addition
    parser.add_option('--r-layers', dest='regression_layers', default=0,
                      help='Number of layers in regression [0|1|2|>2]: default=%default')
    parser.add_option('--r-units', dest='regression_units', default=0,
                      help='Width of layers in regression [0|1|2|>2]: default=%default')
    parser.add_option('--task', dest='task', default="reg",
                      help='Select downstream task: either "class" or "reg".')
    parser.add_option('--reg-intercept', action="store_true", dest='reg_intercept', default=False,
                      help='Whether final regression layer has a bias term.')
    parser.add_option('--no-td-shuffle', action="store_false", dest='train_dev_shuffle', default=True,
                      help='Whether to shuffle the training data for the train-dev split.')
    parser.add_option('--eval-on-fly', action="store_true", dest='test_on_the_fly', default=False,
                      help='Test immediately with best validation model.')
    parser.add_option('--no-train-eval', action="store_false", dest='train_eval', default=True,
                      help='Additionally, evaluate performance on training set.')
    parser.add_option('--dev-prefix', dest='dev_prefix', default=None,
                      help='Prefix of dev set: default=%default')
    parser.add_option('--dev-batch-size', dest='dev_batch_size', default= None,
                      help='Batch size on dev and test set')
    parser.add_option('--test-batch-size', dest='test_batch_size', default= None,
                      help='Batch size on test set')
    parser.add_option('--recent-saves', action="store_true", dest='recent_saves', default=False,
                      help='Save model every 10 epochs.')
    parser.add_option('--eval-last-epoch', action="store_true", dest='eval_last_epoch', default=False,
                      help='Evaluate based on last training epoch.')



    (options, args) = parser.parse_args()

    input_dir = args[0]

    train_prefix = options.train_prefix
    alpha = float(options.alpha)
    n_topics = int(options.n_topics)
    batch_size = int(options.batch_size)
    learning_rate = float(options.learning_rate)
    adam_beta1 = float(options.momentum)
    n_epochs = int(options.epochs)
    encoder_layers = int(options.encoder_layers)
    embedding_dim = int(options.embedding_dim)
    encoder_shortcuts = options.encoder_shortcuts
    label_file_name = options.label_name
    covar_file_names = options.covar_names
    unsup_covar_file_names = options.covar_names
    use_covar_interactions = options.covar_interactions
    infer_covars = options.infer_covars
    label_emb_dim = int(options.label_emb_dim)
    covar_emb_dim = int(options.covar_emb_dim)
    min_covar_count = options.min_covar_count
    classifier_layers = int(options.classifier_layers)
    covars_in_downstream_task = not options.exclude_covars
    auto_regularize = options.regularize
    test_prefix = options.test_prefix
    output_dir = options.output_dir
    word2vec_file = options.word2vec_file
    vocab_size = options.vocab_size
    update_background = options.update_bg
    no_bg = options.no_bg
    bn_anneal = not options.no_bn_anneal
    test_samples = int(options.test_samples)
    dev_folds = int(options.dev_folds)
    dev_fold = int(options.dev_fold)
    optimizer = options.optimizer
    seed = options.seed
    regression_layers = int(options.regression_layers)
    regression_units = int(options.regression_units)
    task = options.task
    reg_intercept = options.reg_intercept
    test_on_the_fly = options.test_on_the_fly
    train_dev_shuffle = options.train_dev_shuffle
    train_eval = options.train_eval
    dev_prefix = options.dev_prefix
    recent_saves = options.recent_saves
    dev_batch_size = options.dev_batch_size
    test_batch_size = options.test_batch_size
    eval_last_epoch = options.eval_last_epoch
    
    # check validity of boolean inputs
    if isinstance(test_on_the_fly, bool) == False:
        raise ValueError("value for eval-on-fly not allowed.")
    if isinstance(train_dev_shuffle, bool) == False:
        raise ValueError("value for td_shuffle not allowed.")
    if isinstance(reg_intercept, bool) == False:
        raise ValueError("value for reg_intercept not allowed.")
    
    
    threads = int(options.threads)
    if seed is not None:
        seed = int(seed)
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState(np.random.randint(0, 100000))

    # load the training data
    train_X, vocab, train_labels, label_names, label_type, train_covariates, covariate_names, covariates_type, col_sel = load_data(input_dir, train_prefix, label_file_name, covar_file_names, vocab_size=vocab_size)
    n_train, dv = train_X.shape
    
    if train_X.shape[0] < batch_size:
        print("\n\n\n\n##########")
        print("Batch-size overwritten. Training sample size is {}. Original batch size was {}.".format(train_X.shape[0],batch_size))
        batch_size = train_X.shape[0]
        print("New batch size is:", batch_size)
        print("\n\n\n\n##########")
        
    if train_labels is not None:
        _, n_labels = train_labels.shape
    else:
        n_labels = 0

    if train_covariates is not None:
        _, n_covariates = train_covariates.shape
        # filter on covariate frequency, if desired
        if min_covar_count is not None and int(min_covar_count) > 0:
            print("Removing rare covariates")
            covar_sums = train_covariates.sum(axis=0).reshape((n_covariates, ))
            covariate_selector = covar_sums > int(min_covar_count)
            train_covariates = train_covariates[:, covariate_selector]
            covariate_names = [name for i, name in enumerate(covariate_names) if covariate_selector[i]]
            n_covariates = len(covariate_names)
    else:
        n_covariates = 0

    # split training data into train and dev
    if dev_folds > 0 and dev_prefix is None:
        n_dev = int(n_train / dev_folds)
        indices = np.array(range(n_train), dtype=int)
        if train_dev_shuffle:
            rng.shuffle(indices)
        if dev_fold < dev_folds - 1:
            dev_indices = indices[n_dev * dev_fold: n_dev * (dev_fold +1)]
        else:
            dev_indices = indices[n_dev * dev_fold:]
        train_indices = list(set(indices) - set(dev_indices))
        dev_X = train_X[dev_indices, :]
        train_X = train_X[train_indices, :]
        if train_labels is not None:
            dev_labels = train_labels[dev_indices, :]
            train_labels = train_labels[train_indices, :]
        else:
            dev_labels = None
        if train_covariates is not None:
            dev_covariates = train_covariates[dev_indices, :]
            train_covariates = train_covariates[train_indices, :]
        else:
            dev_covariates = None
        n_train = len(train_indices)
    else:
        dev_X = None
        dev_labels = None
        dev_covariates = None
        n_dev = 0
        
    # if specific dev set is provided, use this instead of splitting training set
    if dev_prefix is not None: 
        dev_X, _, dev_labels, _, _, dev_covariates, _, _, _ = load_data(input_dir, dev_prefix, label_file_name, covar_file_names, vocab=vocab, col_sel=col_sel)
        n_dev = dev_X.shape[0]
        
        
    # load test data using the same vocabulary
    if test_prefix is not None:
        test_X, _, test_labels, _, _, test_covariates, _, _, _ = load_data(input_dir, test_prefix, label_file_name, covar_file_names, vocab=vocab, col_sel=col_sel)
        n_test, _ = test_X.shape
        if test_labels is not None:
            _, n_labels_test = test_labels.shape
            assert n_labels_test == n_labels
        if test_covariates is not None:
            if min_covar_count is not None and int(min_covar_count) > 0:
                test_covariates = test_covariates[:, covariate_selector]
            _, n_covariates_test = test_covariates.shape
            assert n_covariates_test == n_covariates
        
        
    else:
        test_X = None
        n_test = 0
        test_labels = None
        test_covariates = None
        
    # setup for dev and test batch-sizes                
    if test_batch_size is not None:
        test_batch_size = int(test_batch_size)    
    
    if dev_X is not None:
        if dev_batch_size is not None:
            dev_batch_size = int(dev_batch_size)
        else:
            dev_batch_size = dev_X.shape[0]
        if dev_X.shape[0] < dev_batch_size:
            dev_batch_size = dev_X.shape[0]
    
        
    if test_X is not None:
        if test_batch_size is not None:
            test_batch_size = int(test_batch_size)
        else:
            test_batch_size = test_X.shape[0]
        if test_X.shape[0] < test_batch_size:
            test_batch_size = test_X.shape[0]


    # initialize the background using the overall frequency of terms
    init_bg = get_init_bg(train_X)
    init_beta = None
    update_beta = True
    if no_bg:
        if n_topics == 1:
            init_beta = init_bg.copy()
            init_beta = init_beta.reshape([1, len(vocab)])
            update_beta = False
        init_bg = np.zeros_like(init_bg)

    # create the network configuration
    network_architecture = make_network(dv, encoder_layers, embedding_dim,
                                        n_topics, encoder_shortcuts, label_type, n_labels, label_emb_dim,
                                        covariates_type, n_covariates, covar_emb_dim, use_covar_interactions,
                                        classifier_layers, covars_in_downstream_task, regression_layers, task=task, reg_intercept=reg_intercept,
                                       test_on_the_fly = test_on_the_fly, train_eval = train_eval, recent_saves = recent_saves, eval_last_epoch = eval_last_epoch,
                                       regression_units = regression_units)  # make_network()

    print("Network architecture:")
    for key, val in network_architecture.items():
        print(key + ':', val)
    print("train-val-set shuffle:", train_dev_shuffle)
    print("regularize:", auto_regularize)
    print("shape training corpus:", train_X.shape)
    if dev_X is not None:
        print("shape dev corpus:", dev_X.shape)
        print("dev_batch_size:",dev_batch_size)
    else:
        print("shape dev corpus:", "no dev set")
    if test_X is not None:
        print("shape test corpus:", test_X.shape)
        print("test_batch_size:",test_batch_size)
    else:
        print("shape test corpus:", "no test set")


    # load pretrained word vectors
    if word2vec_file is not None:
        vocab_size = len(
            vocab)
        vocab_dict = dict(zip(vocab, range(vocab_size)))
        embeddings = np.array(rng.rand(vocab_size, embedding_dim) * 0.25 - 0.5, dtype=np.float32)
        count = 0
        print("Loading word vectors")
        if word2vec_file[-3:] == 'bin':
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            pretrained = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=False)

        for word, index in vocab_dict.items():
            if word in pretrained:
                count += 1
                embeddings[index, :] = pretrained[word]

        print("Found embeddings for %d words" % count)
        update_embeddings = False
    else:
        embeddings = None
        update_embeddings = True

    tf.reset_default_graph()

    # create the model
    model = Scholar(network_architecture, alpha=alpha, learning_rate=learning_rate, batch_size=batch_size, init_embeddings=embeddings, update_embeddings=update_embeddings, init_bg=init_bg, update_background=update_background, init_beta=init_beta, update_beta=update_beta, threads=threads, regularize=auto_regularize, optimizer=optimizer, adam_beta1=adam_beta1, seed=seed, output_dir = output_dir)

    # setup saving options
    model.save_model_setup()
    
    # train the model
    print("Optimizing full model")
    model = train(model, network_architecture, train_X, train_labels, train_covariates, regularize=auto_regularize, training_epochs=n_epochs, batch_size=batch_size, rng=rng, X_dev=dev_X, Y_dev=dev_labels, C_dev=dev_covariates, bn_anneal=bn_anneal, output_dir = output_dir, X_test = test_X, Y_test = test_labels, C_test = test_covariates, dev_batch_size = dev_batch_size, test_batch_size = test_batch_size, vocab = vocab, covariate_names = covariate_names, no_bg = no_bg)

    # create output directory
    fh.makedirs(output_dir)
    
    # save vocab
    fh.write_to_json(vocab, os.path.join(output_dir, 'vocab.json'), sort_keys=False)
    
    # evaluate accuracy/mse on labels
    if test_labels is not None and eval_last_epoch == True:
        print("\n###\nModel evaluation - test set\n###\n")
        #print("Predicting labels")
        _, test_task_loss, _, _, _, _ = evaluate(model=model, network_architecture=network_architecture,
                                                 X=test_X, Y=test_labels, C=test_covariates, 
                                                 regularize=auto_regularize, bn_anneal=bn_anneal, 
                                                 batch_size = test_batch_size, output_dir = output_dir, 
                                                 subset ="test_last_epoch", task = task, save_results = True)
        # save latent variables
        save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                     use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                     subset = "test_last_epoch", verbose = True)
        

    if test_labels is None or eval_last_epoch == True:
        # evaluate accuracy on covariates (if categorical)
        if n_covariates > 0 and covariates_type == 'categorical' and infer_covars:
            print("Predicting categorical covariates")
            predictions = infer_categorical_covariate(model, network_architecture, train_X, train_labels)
            accuracy = float(np.sum(predictions == np.argmax(train_covariates, axis=1)) / float(len(train_covariates)))
            print("Train accuracy on covariates = %0.4f" % accuracy)
            if output_dir is not None:
                fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.train.txt'))

            if dev_X is not None:
                predictions = infer_categorical_covariate(model, network_architecture, dev_X, dev_labels)
                accuracy = float(np.sum(predictions == np.argmax(dev_covariates, axis=1)) / float(len(dev_covariates)))
                print("Dev accuracy on covariates = %0.4f" % accuracy)
                if output_dir is not None:
                    fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.dev.txt'))

            if test_X is not None:
                predictions = infer_categorical_covariate(model, network_architecture, test_X, test_labels)
                accuracy = float(np.sum(predictions == np.argmax(test_covariates, axis=1)) / float(len(test_covariates)))
                print("Test accuracy on covariates = %0.4f" % accuracy)
                if output_dir is not None:
                    fh.write_list_to_text([str(accuracy)], os.path.join(output_dir, 'accuracy.test.txt'))


        # Print associations between topics and labels
        if n_labels > 0 and task == "class":
            all_probs = np.zeros([n_topics, n_labels])
            if n_labels < 15:
                print("Label probabilities based on topics")
                print("Labels:", ' '.join([name for name in label_names]))
            for k in range(n_topics):
                Z = np.zeros([1, n_topics]).astype('float32')
                Z[0, k] = 1.0
                if n_covariates > 0:
                    C = np.zeros([1, n_covariates]).astype('float32')
                else:
                    C = None
                probs = model.predict_from_topics(Z, C)
                all_probs[k, :] = probs
                if n_labels < 15:
                    output = str(k) + ': '
                    for i in range(n_labels):
                        output += '%.4f ' % probs[0, i]
                    print(output)
            np.savez(os.path.join(output_dir, 'topic_label_probs.npz'), probs=all_probs)

            if n_covariates > 0:
                all_probs = np.zeros([n_covariates, n_topics])
                for k in range(n_topics):
                    Z = np.zeros([1, n_topics]).astype('float32')
                    Z[0, k] = 1.0
                    Y = None
                    for c in range(n_covariates):
                        C = np.zeros([1, n_covariates]).astype('float32')
                        C[0, c] = 1.0
                        probs = model.predict_from_topics(Z, C)
                        all_probs[c, k] = probs[0, 0]
                np.savez(os.path.join(output_dir, 'covar_topic_probs.npz'), probs=all_probs)

        # save document representations
        theta = model.compute_theta(train_X, train_labels, train_covariates)
        np.savez(os.path.join(output_dir, 'theta.train_last_epoch.npz'), theta=theta)

        if dev_X is not None:
            dev_Y = np.zeros_like(dev_labels)
            theta = model.compute_theta(dev_X, dev_Y, dev_covariates)
            np.savez(os.path.join(output_dir, 'theta.dev_last_epoch.npz'), theta=theta)

        if n_test > 0:
            test_Y = np.zeros_like(test_labels)
            theta = model.compute_theta(test_X, test_Y, test_covariates)
            np.savez(os.path.join(output_dir, 'theta.test_last_epoch.npz'), theta=theta)
        
    # calculate overall compute time
    end_time = datetime.now()
    compute_time = end_time - start_time
    print("computational time", compute_time)
    fh.write_list_to_text([str(compute_time)], os.path.join(output_dir, 'compute_time.txt'))


def load_data(input_dir, input_prefix, label_file_name=None, covar_file_names=None, vocab_size=None, vocab=None, col_sel=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    n_items, temp_size = temp.shape
    print("Loaded %d documents with %d features" % (n_items, temp_size))

    if vocab is None:
        col_sel = None
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
        # filter vocabulary by word frequency
        if vocab_size is not None:
            print("Filtering vocabulary to the most common %d terms" % int(vocab_size))
            col_sums = np.array(temp.sum(axis=0)).reshape((len(vocab), ))
            order = list(np.argsort(col_sums))
            order.reverse()
            col_sel = np.array(np.zeros(len(vocab)), dtype=bool)
            for i in range(int(vocab_size)):
                col_sel[order[i]] = True
            temp = temp[:, col_sel]
            vocab = [word for i, word in enumerate(vocab) if col_sel[i]]

    elif col_sel is not None:
        print("Using given vocabulary")
        temp = temp[:, col_sel]

    X = np.array(temp, dtype='float32')
    n_items, vocab_size = X.shape
    assert vocab_size == len(vocab)
    print("Loaded %d documents with %d features" % (n_items, vocab_size))

    # filter out empty documents
    non_empty_sel = X.sum(axis=1) > 0
    print("Found %d non-empty documents" % np.sum(non_empty_sel))
    X = X[non_empty_sel, :]
    n_items, vocab_size = X.shape

    if label_file_name is not None:
        label_file = os.path.join(input_dir, input_prefix + '.' + label_file_name + '.csv')
        if os.path.exists(label_file):
            print("Loading labels from", label_file)
            temp = pd.read_csv(label_file, header=0, index_col=0)
            label_names = temp.columns
            labels = np.array(temp.values)
            labels = labels[non_empty_sel, :]
            n, n_labels = labels.shape
            assert n == n_items
            print("%d labels" % n_labels)
        else:
            print("Label file not found:", label_file)
            sys.exit()
        if (np.sum(labels, axis=1) == 1).all() and (np.sum(labels == 0) + np.sum(labels == 1) == labels.size):
            label_type = 'categorical'
        elif np.sum(labels == 0) + np.sum(labels == 1) == labels.size:
            label_type = 'bernoulli'
        else:
            label_type = 'real'
        print("Found labels of type %s" % label_type)

    else:
        labels = None
        label_names = None
        label_type = None

    if covar_file_names is not None:
        covariate_list = []
        covariate_names_list = []
        covar_file_names = covar_file_names.split(',')
        for covar_file_name in covar_file_names:
            covariates_file = os.path.join(input_dir, input_prefix + '.' + covar_file_name + '.csv')
            if os.path.exists(covariates_file):
                print("Loading covariates from", covariates_file)
                temp = pd.read_csv(covariates_file, header=0, index_col=0)
                covariate_names = temp.columns
                covariates = np.array(temp.values, dtype=np.float32)
                covariates = covariates[non_empty_sel, :]
                n, n_covariates = covariates.shape
                assert n == n_items
                covariate_list.append(covariates)
                covariate_names_list.extend(covariate_names)
            else:
                print("Covariates file not found:", covariates_file)
                sys.exit()
        covariates = np.hstack(covariate_list)
        covariate_names = covariate_names_list
        n, n_covariates = covariates.shape

        if (np.sum(covariates, axis=1) == 1).all() and (np.sum(covariates == 0) + np.sum(covariates == 1) == covariates.size):
            covariates_type = 'categorical'
        else:
            covariates_type = 'other'

        print("Found covariates of type %s" % covariates_type)

        assert n == n_items
        print("%d covariates" % n_covariates)
    else:
        covariates = None
        covariate_names = None
        covariates_type = None

    counts_sum = X.sum(axis=0)
    order = list(np.argsort(counts_sum).tolist())
    order.reverse()
    print("Most common words: ", ' '.join([vocab[i] for i in order[:10]]))

    return X, vocab, labels, label_names, label_type, covariates, covariate_names, covariates_type, col_sel


def get_init_bg(data):
    """
    Compute the log background frequency of all words
    """
    sums = np.sum(data, axis=0)+1
    print("Computing background frequencies")
    print("Min/max word counts in training data: %d %d" % (int(np.min(sums)), int(np.max(sums))))
    bg = np.array(np.log(sums) - np.log(float(np.sum(sums))), dtype=np.float32)
    return bg


def create_minibatch(X, Y, C, batch_size=200, rng=None):
    """
    Split data into minibatches
    """
    while True:
        # Return random data samples of a size 'minibatch_size' at each iteration
        if rng is not None:
            ixs = rng.randint(X.shape[0], size=batch_size)
        else:
            ixs = np.random.randint(X.shape[0], size=batch_size)
        if Y is not None and C is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), C[ixs, :].astype('float32')
        elif Y is not None:
            yield X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), None
        elif C is not None:
            yield X[ixs, :].astype('float32'), None, C[ixs, :].astype('float32')
        else:
            yield X[ixs, :].astype('float32'), None, None

            

def make_network(dv, encoder_layers=2, embedding_dim=300, n_topics=50, encoder_shortcut=False, label_type=None, n_labels=0, label_emb_dim=0, covariate_type=None, n_covariates=0, covar_emb_dim=0, use_covar_interactions=False, classifier_layers=1, covars_in_downstream_task=True, regression_layers = 0, task = "reg", reg_intercept=True, test_on_the_fly = False, train_eval = False, recent_saves = False, eval_last_epoch = False, regression_units = 0):
    """
    Combine the network configuration parameters into a dictionary
    """
    tf.reset_default_graph()
    network_architecture = \
        dict(encoder_layers=encoder_layers,
             encoder_shortcut=encoder_shortcut,
             embedding_dim=embedding_dim,
             n_topics=n_topics,
             dv=dv,
             label_type=label_type,
             n_labels=n_labels,
             label_emb_dim=label_emb_dim,
             covariate_type=covariate_type,
             n_covariates=n_covariates,
             covar_emb_dim=covar_emb_dim,
             use_covar_interactions=use_covar_interactions,
             classifier_layers=classifier_layers,
             regression_layers=regression_layers,
             covars_in_downstream_task=covars_in_downstream_task,
             task=task,
             reg_intercept = reg_intercept,
             test_on_the_fly = test_on_the_fly,
             train_eval = train_eval,
             recent_saves = recent_saves,
             eval_last_epoch = eval_last_epoch,
             regression_units = regression_units
             )
    return network_architecture


def save_latent_vars(model, task, vocab, n_covariates, covariate_names, use_covar_interactions, 
                     covar_emb_dim, no_bg, output_dir, subset = None, verbose = False):    
    # get regression weights
    if task =="reg":
        W, b = model.get_reg_weights()
        pd.DataFrame(data = W).to_csv(os.path.join(output_dir, 'regression_weights.'+subset+'.csv'))
        pd.DataFrame(data = b).to_csv(os.path.join(output_dir, 'regression_bias.'+subset+'.csv'))

    # print background
    bg = model.get_bg()
    if not no_bg and verbose == True:
        print_top_bg(bg, vocab)

    # print topics
    emb = model.get_weights()
    maw, sparsity = print_top_words(emb, vocab, verbose = verbose)
    save_weights(output_dir, emb, bg, vocab, sparsity_threshold=1e-5, subset = subset)
    if verbose:
        print("Topics:")
        print("sparsity in topics = %0.4f" % sparsity)

    fh.write_list_to_text(['{:.4f}'.format(maw)], os.path.join(output_dir, 'maw.'+subset+'.txt'))
    fh.write_list_to_text(['{:.4f}'.format(sparsity)], os.path.join(output_dir, 'sparsity.'+subset+'.txt'))

    if n_covariates > 0:
        beta_c = model.get_covar_weights()
        if verbose: print("Covariate deviations:")
        if covar_emb_dim > 0:
            maw, sparsity = print_top_words(beta_c, vocab, verbose = verbose)
        else:
            maw, sparsity = print_top_words(beta_c, vocab, covariate_names, verbose = verbose)
        if verbose: print("sparsity in covariates = %0.4f" % sparsity)
        if output_dir is not None:
            np.savez(os.path.join(output_dir, 'beta_c.'+subset+'.npz'), beta=beta_c, names=covariate_names)
        if use_covar_interactions:
            if verbose: print("Covariate interactions")
            beta_ci = model.get_covar_inter_weights()
            if verbose: print(beta_ci.shape)
            if covariate_names is not None:
                names = [str(k) + ':' + c for k in range(n_topics) for c in covariate_names]
            else:
                names = None
            maw, sparsity = print_top_words(beta_ci, vocab, names, verbose = verbose)
            if output_dir is not None:
                np.savez(os.path.join(output_dir, 'beta_ci.'+subset+'.npz'), beta=beta_ci, names=names)
            if verbose:
                print("sparsity in covariate interactions = %0.4f" % sparsity)
                print("Combined covariates and interactions:")

        if covar_emb_dim > 0:
            print_covariate_embeddings(model, covariate_names, output_dir, subset = subset, verbose = verbose)


def train(model, network_architecture, X, Y, C, batch_size=200, training_epochs=100, display_step=1, 
          min_weights_sq=1e-7, regularize=False, X_dev=None, Y_dev=None, C_dev=None, bn_anneal=True, 
          init_eta_bn_prop=1.0, rng=None, output_dir = None, verbose_updates = False,
          X_test = None, Y_test = None, C_test = None, dev_batch_size = 1, test_batch_size = 1,
          vocab = None, covariate_names = None, no_bg = None):

    n_train, dv = X.shape
    mb_gen = create_minibatch(X, Y, C, batch_size=batch_size, rng=rng)

    dv = network_architecture['dv']
    n_topics = network_architecture['n_topics']
    task = network_architecture['task']
    test_on_the_fly = network_architecture['test_on_the_fly']
    train_eval = network_architecture['train_eval']
    recent_saves = network_architecture['recent_saves']
    n_covariates = network_architecture['n_covariates']
    use_covar_interactions = network_architecture['use_covar_interactions']
    covar_emb_dim = network_architecture['covar_emb_dim']

    total_batch = int(n_train / batch_size)

    # create np arrays to store regularization strengths, which we'll update outside of the tensorflow model
    if regularize:
        l2_strengths = 0.5 * np.ones([n_topics, dv]) / float(n_train)
        l2_strengths_c = 0.5 * np.ones([model.beta_c_length, dv]) / float(n_train)
        l2_strengths_ci = 0.5 * np.ones([model.beta_ci_length, dv]) / float(n_train)
    else:
        l2_strengths = np.zeros([n_topics, dv])
        l2_strengths_c = np.zeros([model.beta_c_length, dv])
        l2_strengths_ci = np.zeros([model.beta_ci_length, dv])
        
    # for training set evaluation
    if task == "reg":
        task_name = "mse"
        best_train_loss = np.inf
    else:
        task_name = "accuracy"
        best_train_loss = 0.0

    # for dev set evaluation
    if X_dev is not None:
        if task == "reg":
            best_loss = np.inf
        else:
            best_loss = 0.0
        best_plxy = np.inf

    # epoch trackers
    epoch_tracker_task_loss = []
    epoch_tracker_dev_task_loss = []
    epoch_tracker_test_task_loss = []
    epoch_tracker_plxy = []
    epoch_tracker_dev_plxy = []
    epoch_tracker_test_plxy = []
    
    eta_bn_prop = init_eta_bn_prop  # interpolation between batch norm and no batch norm in final layer of recon
    kld_weight = 1.0  # could use this to anneal KLD, but not currently doing so

    # Training cycle
    for epoch in range(training_epochs):
        avg_loss = 0.
        avg_task_loss = 0.
        task_loss = 0.
        if task == "class":
            task_name = "accuracy"
        elif task == "reg":
            task_name = "mse"
        # Loop over all batches
        for i in range(total_batch):
            # get a minibatch
            batch_xs, batch_ys, batch_cs = next(mb_gen)
            # do one update, passing in the data, regularization strengths, and bn
            loss, task_loss_i, pred = model.fit(batch_xs, batch_ys, batch_cs, l2_strengths=l2_strengths, 
                                             l2_strengths_c=l2_strengths_c, l2_strengths_ci=l2_strengths_ci,
                                             eta_bn_prop=eta_bn_prop, kld_weight=kld_weight)
            # compute accuracy/mse on minibatch
            if network_architecture['n_labels'] > 0 and task == "class":
                task_loss += np.sum(pred == np.argmax(batch_ys, axis=1)) / float(n_train) # accuracy
            elif network_architecture['n_labels'] > 0 and task == "reg":
                task_loss += np.sum((batch_ys - pred)**2) / float(n_train) # mse
            # Compute average loss
            avg_loss += loss / n_train * batch_size
            avg_task_loss += task_loss_i / n_train * batch_size
            # Save predictions
            if Y is not None:
                if i==0:
                    train_predictions = pred
                else:
                    train_predictions = np.append(train_predictions,pred,axis=0)
            if np.isnan(avg_loss):
                print(epoch, i, np.sum(batch_xs, 1).astype(np.int), batch_xs.shape)
                print('Encountered NaN, stopping training. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()
        # log task loss for this epoch
        epoch_tracker_task_loss.append(task_loss)
                
        # evaluate on training set if no dev set available (only use for small training samples where no dev set is available)
        if X_dev is None and network_architecture['n_labels'] > 0 and train_eval==True:
            if task == "class":
                if  task_loss > best_train_loss: #accuracy
                    best_train_loss = task_loss
                    print("training | epoch: {} | new best training {}: {:.4}".format(epoch, task_name, best_train_loss))
                    model.best_acc_saver.save(model.sess, model.checkpoint_dir + '/best-model-train_acc={:g}-epoch{}.ckpt'.format(best_train_loss, epoch))
                    _, _, _, _, _, train_plxy_text = evaluate(model=model, 
                                            network_architecture=network_architecture,
                                            X=X, Y=Y, C=C,
                                            regularize=regularize, bn_anneal=0.0, 
                                            batch_size = X.shape[0], 
                                            output_dir = output_dir,
                                            subset ="best_train", task = task, 
                                            save_results=True)
                    # save latent variables
                    save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                                     use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                                     subset = "best_train", verbose = False)
                    epoch_tracker_plxy.append(train_plxy_text)
                else:
                    print("training | epoch: {0} | last training {1} = {2}; best training {1} = {3} ".format(
                        epoch, task_name, round(task_loss,4), round(best_train_loss,4)))
            elif task == "reg":
                if task_loss < best_train_loss: #mse
                    best_train_loss = task_loss
                    print("training | epoch: {} | new best training {}: {:.4}".format(epoch, task_name, best_train_loss))
                    model.best_mse_saver.save(model.sess, model.checkpoint_dir + '/best-model-train_mse={:g}-epoch{}.ckpt'.format(best_train_loss, epoch))
                    # save regression coefficients
                    #W, b = model.get_reg_weights()
                    #pd.DataFrame(data = W).to_csv(os.path.join(output_dir, 'regression_weights.best_train.csv'))
                    #pd.DataFrame(data = b).to_csv(os.path.join(output_dir, 'regression_bias.best_train.csv'))
                    _, _, _, _, _, train_plxy_text = evaluate(model=model, 
                                            network_architecture=network_architecture,
                                            X=X, Y=Y, C=C,
                                            regularize=regularize, bn_anneal=0.0, 
                                            batch_size = X.shape[0], 
                                            output_dir = output_dir,
                                            subset ="best_train", task = task, 
                                            save_results=True)
                    # save latent variables
                    save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                                     use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                                     subset = "best_train", verbose = False)
                    epoch_tracker_plxy.append(train_plxy_text)
                    if test_on_the_fly:
                        _, test_task_loss, _, _, _,test_plxy_text  = evaluate(model=model, 
                                                                            network_architecture=network_architecture,
                                                                            X=X_test, Y=Y_test, C=C_test,
                                                                            regularize=regularize, bn_anneal=0.0, 
                                                                            batch_size = test_batch_size, 
                                                                            output_dir = output_dir,
                                                                            subset ="test_best_dev", task = task, 
                                                                            save_results=True)
                        epoch_tracker_test_task_loss.append(test_task_loss)
                        epoch_tracker_test_plxy.append(test_plxy_text)
                else:
                    print("training | epoch: {0} | last training {1} = {2}; best training {1} = {3} ".format(
                        epoch, task_name, round(task_loss,4), round(best_train_loss,4)))
        
         

        # evaluate on dev set
        if X_dev is not None and network_architecture['n_labels'] > 0:
            dev_predictions, dev_task_loss, _, _, _, dev_plxy_text = evaluate(model = model, network_architecture = network_architecture, 
                                                                                                        X=X_dev, Y=Y_dev, C=C_dev, 
                                                                                                        regularize=regularize, bn_anneal=0.0, 
                                                                                                        batch_size = dev_batch_size, output_dir = output_dir, 
                                                                                                        subset ="dev_eval", task = task, save_results = False)
            epoch_tracker_dev_task_loss.append(dev_task_loss)
            epoch_tracker_dev_plxy.append(dev_plxy_text)
            if task == "class":
                if dev_task_loss > best_loss: #accuracy
                    best_loss = dev_task_loss
                    print("validation | epoch: {} | new best validation {}: {:.4}".format(epoch, task_name,best_loss))
                    model.best_acc_saver.save(model.sess, model.checkpoint_dir + '/best-model-val_acc={:g}-epoch{}.ckpt'.format(best_loss, epoch))
                    # save latent variables
                    save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                                     use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                                     subset = "best_dev", verbose = False)
                    if train_eval:
                        _, _, _, _, _, train_plxy_text = evaluate(model=model, 
                                                network_architecture=network_architecture,
                                                X=X, Y=Y, C=C,
                                                regularize=regularize, bn_anneal=0.0, 
                                                batch_size = X.shape[0], 
                                                output_dir = output_dir,
                                                subset ="train_best_dev", task = task, 
                                                save_results=True)

                        epoch_tracker_plxy.append(train_plxy_text)
                else:
                    print("validation | epoch: {0} | last validation {1} = {2}; best validation {1} = {3} ".format(
                        epoch, task_name, round(dev_task_loss,4), round(dev_task_loss,4)))
                    
            elif task == "reg":
                if dev_task_loss < best_loss: #mse
                    best_loss = dev_task_loss
                    print("validation | epoch: {} | new best validation {}: {:.4}".format(epoch, task_name,best_loss))
                    model.best_mse_saver.save(model.sess, model.checkpoint_dir + '/best-model-val_mse={:g}-epoch{}.ckpt'.format(best_loss, epoch))
                    # save regression coefficients
                    #W, b = model.get_reg_weights()
                    #pd.DataFrame(data = W).to_csv(os.path.join(output_dir, 'regression_weights.best_dev.csv'))
                    #pd.DataFrame(data = b).to_csv(os.path.join(output_dir, 'regression_bias.best_dev.csv'))
                    # save latent variables
                    save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                                     use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                                     subset = "best_dev", verbose = False)
                    if train_eval:
                        _, _, _, _, _, train_plxy_text = evaluate(model=model, 
                                                    network_architecture=network_architecture,
                                                    X=X, Y=Y, C=C,
                                                    regularize=regularize, bn_anneal=0.0, 
                                                    batch_size = X.shape[0], 
                                                    output_dir = output_dir,
                                                    subset ="train_best_dev", task = task, 
                                                    save_results=True)
                        epoch_tracker_plxy.append(train_plxy_text)
                    if test_on_the_fly:
                        _, test_task_loss, _, _, _, test_plxy_text = evaluate(model=model, 
                                                                network_architecture=network_architecture,
                                                                X=X_test, Y=Y_test, C=C_test,
                                                                regularize=regularize, bn_anneal=0.0, 
                                                                batch_size = test_batch_size,
                                                                output_dir = output_dir,
                                                                subset ="test_best_dev", task = task,
                                                                save_results=True)
                        epoch_tracker_test_task_loss.append(test_task_loss)
                        epoch_tracker_test_plxy.append(test_plxy_text)
                else:
                    print("validation | epoch: {0} | last validation {1} = {2}; best validation {1} = {3} ".format(
                        epoch, task_name, round(dev_task_loss,4), round(best_loss,4)))

            print("validation | epoch: {} | dev perplexity = {:.4f}".format(epoch, dev_plxy_text))
            if verbose_updates:
                print("validation | y_act (first 3):\n", Y_dev[:3])
                print("validation | y_pred (first 3):\n", dev_predictions[:3])

        # if unsupervised, use model that performs best in dev set based on perplexity
        if X_dev is not None and network_architecture['n_labels'] == 0:
            _, _, dev_avg_loss, _, _, dev_plxy_text = evaluate(model = model, network_architecture = network_architecture, 
                                                                                            X=X_dev, Y=Y_dev, C=C_dev, 
                                                                                            regularize=regularize, bn_anneal=0.0, 
                                                                                            batch_size = dev_batch_size, output_dir = output_dir, 
                                                                                            subset ="dev_eval", task = task, save_results = False)
            epoch_tracker_dev_plxy.append(dev_plxy_text)
            if dev_plxy_text < best_plxy: #perplexity on text only
                best_plxy = dev_plxy_text
                print("validation | epoch: {} | new best validation perplexity: {:.4}".format(epoch, best_plxy))
                model.best_plxy_saver.save(model.sess, model.checkpoint_dir + '/best-model-val_mse={:g}-epoch{}.ckpt'.format(best_plxy, epoch))
                if X_test is not None:
                        # estimate thetas
                                # save document representations
                                #theta = model.compute_theta(X_test, Y_test, Y_test)
                                #np.savez(os.path.join(output_dir, 'theta.train_last_epoch.npz'), theta=theta)
                        _, _, _, _, _, test_plxy_text = evaluate(model = model, network_architecture = network_architecture, 
                                                                                            X=X_test, Y=None, C= C_test, 
                                                                                            regularize=regularize, bn_anneal=0.0, 
                                                                                            batch_size = test_batch_size, output_dir = output_dir, 
                                                                                            subset ="test_best_dev", task = task, save_results = True)
                        epoch_tracker_test_plxy.append(test_plxy_text)
                # save latent variables
                save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                                 use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                                 subset = "best_dev", verbose = False)
                # save training evaluation
                _, _, _, _, _, train_plxy_text = evaluate(model=model, 
                                            network_architecture=network_architecture,
                                            X=X, Y=Y, C=C,
                                            regularize=regularize, bn_anneal=0.0, 
                                            batch_size = X.shape[0], 
                                            output_dir = output_dir,
                                            subset ="train_best_dev", task = task, 
                                            save_results=True)
                epoch_tracker_plxy.append(train_plxy_text)
            else:
                print("validation | epoch: {0} | last validation {1} = {2}; last validation {3} = {4}; best validation {3} = {5} ".format(
                    epoch, "avg. loss", round(dev_avg_loss,4),"perplexity", round(dev_plxy_text,4), round(best_plxy,4)))

            
        # update weight prior variances using current weight values
        if regularize:
            weights = model.get_weights()
            weights_sq = weights ** 2
            # avoid infinite regularization
            weights_sq[weights_sq < min_weights_sq] = min_weights_sq
            l2_strengths = 0.5 / weights_sq / float(n_train)

            if network_architecture['n_covariates'] > 0:
                weights = model.get_covar_weights()
                weights_sq = weights ** 2
                # avoid infinite regularization
                weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                l2_strengths_c = 0.5 / weights_sq / float(n_train)
                if network_architecture['use_covar_interactions']:
                    weights = model.get_covar_inter_weights()
                    weights_sq = weights ** 2
                    # avoid infinite regularization
                    weights_sq[weights_sq < min_weights_sq] = min_weights_sq
                    l2_strengths_ci = 0.5 / weights_sq / float(n_train)
                    
        # anneal eta_bn_prop from 1 to 0 over the course of training
        if bn_anneal:
            if eta_bn_prop > 0:
                eta_bn_prop -= 1.0 / float(training_epochs*0.75)
                if eta_bn_prop < 0:
                    eta_bn_prop = 0.0

        # Display logs per epoch step
        if epoch % display_step == 0 and epoch > 0:
            if network_architecture['n_labels'] > 0:
                print("training | epoch:", '%d' % epoch, "| loss =", "{:.9f}".format(avg_loss), "; avg_{}_loss =".format(task_name), "{:.9f}".format(avg_task_loss), "; training {} (noisy) =".format(task_name), "{:.9f}".format(task_loss))
            else:
                print("training | epoch:", '%d' % epoch, "loss=", "{:.9f}".format(avg_loss))
        
        # save current model
        if epoch % 10==0 and recent_saves==True:
            model.recent_saver.save(model.sess, model.checkpoint_dir + '/recent-model-val_{}={:g}-epoch{}.ckpt'.format(task_name, dev_task_loss, epoch))

           
    # option to evaluate training set after last training epoch
    if network_architecture['n_labels'] < 1 or network_architecture['eval_last_epoch']==True:
        # evaluate insample training on last epoch
        _, _, _, _, _, _ = evaluate(model=model, 
                                network_architecture=network_architecture,
                                X=X, Y=Y, C=C,
                                regularize=regularize, bn_anneal=0.0, 
                                batch_size = X.shape[0], 
                                output_dir = output_dir,
                                subset ="train_last_epoch", task = task, 
                                save_results=True)
        # save latent variables
        save_latent_vars(model, task, vocab, n_covariates, covariate_names, 
                         use_covar_interactions, covar_emb_dim, no_bg, output_dir, 
                         subset = "train_last_epoch", verbose = False)
    
    # save task_loss trackers
    train_loss_tracker = pd.Series(data = epoch_tracker_task_loss, name = "train_{}_per_epoch".format(task_name))
    dev_loss_tracker = pd.Series(data = epoch_tracker_dev_task_loss, name = "dev_{}_per_epoch".format(task_name))
    test_loss_tracker = pd.Series(data = epoch_tracker_test_task_loss, name = "test_{}_per_epoch".format(task_name))
    train_plxy_tracker = pd.Series(data = epoch_tracker_plxy, name = "train_plxy_per_epoch")
    dev_plxy_tracker = pd.Series(data = epoch_tracker_dev_plxy, name = "dev_plxy_per_epoch")
    test_plxy_tracker = pd.Series(data = epoch_tracker_test_plxy, name = "test_plxy_per_epoch")
    
    if output_dir is not None:
        cols = ["red","green","blue","red","green","blue"]
        tracker_names = ["training_loss","validation_loss","test_loss","training_plxy","validation_plxy","test_plxy"]
        for idx, tracker in enumerate([train_loss_tracker, dev_loss_tracker, test_loss_tracker,
                                       train_plxy_tracker, dev_plxy_tracker, test_plxy_tracker]):
            plt.figure(figsize=(5,5))
            plt.plot(tracker, color = cols[idx], alpha = 0.8)
            plt.title(tracker_names[idx])
            plt.xlabel("epoch")
            if idx < 3:
                plt.ylabel(task_name)
            else:
                 plt.ylabel("text_perplexity")
            plt.savefig(os.path.join(output_dir, "plot."+tracker_names[idx] +'.per_epoch.pdf'))
            plt.close()
            
    return model


def infer_categorical_covariate(model, network_architecture, X, Y, eta_bn_prop=0.0):
    """
    Predict the value of categorical covariates for each instances based on log probability of words
    """
    n_items, vocab_size = X.shape
    n_covariates = network_architecture['n_covariates']
    n_labels = network_architecture['n_labels']
    predictions = np.zeros(n_items, dtype=int)

    if n_covariates == 1:
        for i in range(n_items):
            C_i = np.zeros((2, 1)).astype('float32')
            C_i[1, 0] = 1.0
            X_i = np.zeros((2, vocab_size)).astype('float32')
            X_i[:, :] = X[i, :]
            if Y is not None:
                Y_i = np.zeros((2, n_labels)).astype('float32')
                Y_i[:, :] = Y[i, :]
            else:
                Y_i = None
            losses = model.get_losses(X_i, Y_i, C_i, eta_bn_prop=eta_bn_prop)
            pred = np.argmin(losses)
            predictions[i] = pred

    else:
        # process instances one by one
        for i in range(n_items):
            # create a matrix of all possible covariate values and evaluate all as a minibatch
            C_i = np.eye(n_covariates).astype('float32')
            X_i = np.zeros((n_covariates, vocab_size)).astype('float32')
            X_i[:, :] = X[i, :]
            if Y is not None:
                Y_i = np.zeros((n_covariates, n_labels)).astype('float32')
                Y_i[:, :] = Y[i, :]
            else:
                Y_i = None
            # check the log-loss for each possible value of C and take the best
            losses = model.get_losses(X_i, Y_i, C_i, eta_bn_prop=eta_bn_prop)
            pred = np.argmin(losses)
            predictions[i] = pred

    return predictions


def print_top_words(beta, feature_names, topic_names=None, n_top_words=8, sparsity_threshold=1e-5, values=False, verbose = False):
    """
    Display the highest and lowest weighted words in each topic, along with mean ave weight and sparisty
    """
    sparsity_vals = []
    maw_vals = []
    for i in range(len(beta)):
        # sort the beta weights
        order = list(np.argsort(beta[i]))
        order.reverse()
        output = ''
        # get the top words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        order.reverse()
        output += ' / '
        # get the bottom words
        for j in range(n_top_words):
            if np.abs(beta[i][order[j]]) > sparsity_threshold:
                output += feature_names[order[j]] + ' '
                if values:
                    output += '(' + str(beta[i][order[j]]) + ') '

        # compute sparsity
        sparsity = float(np.sum(np.abs(beta[i]) < sparsity_threshold) / float(len(beta[i])))
        maw = np.mean(np.abs(beta[i]))
        sparsity_vals.append(sparsity)
        maw_vals.append(maw)
        output += ': MAW=%0.4f' % maw + '; sparsity=%0.4f' % sparsity

        # print the topic summary
        if topic_names is not None:
            output = topic_names[i] + ': ' + output
        else:
            output = str(i) + ': ' + output
        if verbose:
            print(output)

    # return mean average weight and sparsity
    return np.mean(maw_vals), np.mean(sparsity_vals)


def print_top_bg(bg, feature_names, n_top_words=10):
    """
    Print the most highly weighted words in the background log frequency
    """
    print('Background frequencies of top words:')
    print(" ".join([feature_names[j]
                    for j in bg.argsort()[:-n_top_words - 1:-1]]))
    temp = bg.copy()
    temp.sort()
    print(np.exp(temp[:-n_top_words-1:-1]))


def evaluate_perplexity(model, X, Y, C, eta_bn_prop=1.0, n_samples=0):
    """
    perplexity_all: Evaluate the approximate perplexity on a subset of the data (using words, labels, and covariates)
    perplexity_gen: Evaluate the approximate perplexity on a subset of the data (using words only)
    """
    # count the number of words in each document
    doc_sums = np.array(X.sum(axis=1), dtype=float)
    X = X.astype('float32')
    if Y is not None:
        Y = Y.astype('float32')
    if C is not None:
        C = C.astype('float32')
    # get the losses for all instances
    all_losses = model.get_losses(X, Y, C, eta_bn_prop=eta_bn_prop, n_samples=n_samples)
    # compute perplexity for all documents in a single batch
    perplexity_all = np.exp(np.mean(all_losses / doc_sums))
    
    # get the generative loss for the document modelling part
    gen_loss = model.get_generative_loss(X, Y, C, eta_bn_prop=eta_bn_prop, n_samples=n_samples)
    # compute perplexity for all documents in a single batch
    perplexity_gen = np.exp(np.mean(gen_loss / doc_sums))

    return perplexity_all, perplexity_gen


def save_weights(output_dir, beta, bg, feature_names, sparsity_threshold=1e-5, subset = None):
    """
    Save model weights to npz files (also the top words in each topic)
    """
    np.savez(os.path.join(output_dir, 'beta.'+subset+'.npz'), beta=beta)
    if bg is not None:
        np.savez(os.path.join(output_dir, 'bg.'+subset+'.npz'), bg=bg)
    topics_file = os.path.join(output_dir, 'topics_sparsity_weighted.'+subset+'.txt')
    topics_file_raw = os.path.join(output_dir, 'topics_unweighted.'+subset+'.txt')
    lines = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words = [feature_names[j] for j in order[:40] if beta[i][j] > sparsity_threshold]
        try:
            output = ' '.join(pos_words)
        except TypeError:
            output = ' '.join(str(pos_words))
        lines.append(output)
     
    lines_raw = []
    for i in range(len(beta)):
        order = list(np.argsort(beta[i]))
        order.reverse()
        pos_words_raw = [feature_names[j] for j in order[:40]]
        try:
            output_raw = ' '.join(pos_words_raw)
        except TypeError:
            output_raw = ' '.join(str(pos_words_raw))    
        lines_raw.append(output_raw)

    fh.write_list_to_text(lines, topics_file) 
    fh.write_list_to_text(lines_raw, topics_file_raw)



def print_label_embeddings(model, class_names):
    """
    Display label embeddings
    """
    label_embeddings = model.get_label_embeddings()
    n_labels, _ = label_embeddings.shape
    dists = np.zeros([n_labels, n_labels])
    for i in range(n_labels):
        for j in range(n_labels):
            emb_i = label_embeddings[i, :]
            emb_j = label_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_labels):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = class_names[i] + ': '
        for j in range(4):
            output += class_names[order[j]] + ' '
        print(output)


def print_covariate_embeddings(model, covariate_names, output_dir, subset = None, verbose = False):
    """
    Display covariate embeddings
    """
    covar_embeddings = model.get_covar_embeddings()
    n_covariates , emb_dim = covar_embeddings.shape
    dists = np.zeros([n_covariates, n_covariates])
    for i in range(n_covariates):
        for j in range(n_covariates):
            emb_i = covar_embeddings[i, :]
            emb_j = covar_embeddings[j, :]
            dists[i, j] = np.dot(emb_i, emb_j) / np.sqrt(np.dot(emb_i, emb_i)) / np.sqrt(np.dot(emb_j, emb_j))
    for i in range(n_covariates):
        order = list(np.argsort(dists[i, :]))
        order.reverse()
        output = covariate_names[i] + ': '
        for j in range(4):
            output += covariate_names[order[j]] + ' '
        print(output)
    if n_covariates < 30 and emb_dim < 10 and verbose == True:
        print(covar_embeddings)
    np.savez(os.path.join(output_dir, 'covar_emb.'+subset+'.npz'), W=covar_embeddings, names=covariate_names)



def predict_labels(model, X, C, Y=None, eta_bn_prop=0.0, task = None):
    """
    Predict a label for each instance using the classifier (or regression) part of the network
    """
    n_items, vocab_size = X.shape
    predictions = np.zeros(n_items, dtype=int)
    task_losses = np.zeros(n_items, dtype=int)
    losses = np.zeros(n_items, dtype=int)
    
    if task=="class":
        print("prediction task: classification")
    elif task =="reg":
        print("prediction task: regression")
    # predict items one by one
    for i in range(n_items):
        X_i = np.expand_dims(X[i, :], axis=0)
        # optionally provide covariates
        if C is not None:
            C_i = np.expand_dims(C[i, :], axis=0)
        else:
            C_i = None
        # predict probabilities
        if task == "reg":
            Y_i = Y[i, :].astype('float32').reshape(1,1)
            pred = model.predict_reg(X=X_i, C=C_i, Y=Y_i, eta_bn_prop=eta_bn_prop,task="reg")
            predictions[i] = pred
            task_losses  = np.nan
            losses = np.nan
            #print("opt:", opt)
            print("y_pred_test (first 3):", predictions[:3])
            return predictions, task_losses, losses


def evaluate_training(model, task, task_loss, X, Y, C, eta_bn_prop, output_dir, train_pred, subset):
    # Training set evaluation
    train_perplexity_all, train_perplexity_text = evaluate_perplexity(model, X, Y, C, eta_bn_prop=eta_bn_prop)
    if output_dir is not None:
        pd.Series(data= train_perplexity_all, name = "perplexity").to_csv(os.path.join(output_dir, subset + '_perplexity_all.csv'))
        pd.Series(data= train_perplexity_text, name = "perplexity").to_csv(os.path.join(output_dir, subset + '_perplexity_text.csv'))
    if task == "reg":
        pR = 1-(task_loss/np.var(Y))
        if output_dir is not None:
            pd.Series(data= task_loss, name = "mse").to_csv(os.path.join(output_dir,'mse.'+subset+'.csv'))
            pd.Series(data= pR, name = "pR").to_csv(os.path.join(output_dir,'pR2.'+subset+'.csv'))
            print("{} | overall mse on labels:".format("train"), task_loss)
            print("{} | overall R^2 om labels:".format("train"), pR)
            y_series = pd.DataFrame(Y)
            pred_series = pd.DataFrame(train_pred)
            # save actuals and predictions
            y_series.to_csv(os.path.join(output_dir, 'y_actuals.'+subset+'.csv'))
            pred_series.to_csv(os.path.join(output_dir,'y_predictions.'+subset+'.csv'))
            # separate targets
            if Y.shape[1] >1:
                target_mses = np.sum((y_series - pred_series)**2) / float(y_series.shape[0])
                target_pRs = 1-(target_mses/np.var(y_series))
                print("{} | target-variance of y:".format("train"), np.var(y_series).values)
                print("{} | target-mse on labels =".format("train"), target_mses.values)
                print("{} | target-R^2 on labels = ".format("train"), target_pRs.values)
                target_mses.to_csv(os.path.join(output_dir, subset + '_mses_per_target.csv'))
                target_pRs.to_csv(os.path.join(output_dir, subset + '_pRs_per_target.csv'))  



def evaluate(model, network_architecture, X, Y, C, display_step= 200, min_weights_sq=1e-7, regularize=False, bn_anneal=True, init_eta_bn_prop=0.0, rng=None, batch_size = 200,output_dir=None, subset=None, task = None, verbose_updates = False, save_results = True):
    """
    Predict a label for each instance using the classifier (or regression) part of the network
    """
    n_items, vocab_size = X.shape
    dv = network_architecture['dv']
    n_topics = network_architecture['n_topics']
    task = network_architecture['task']
    
    # input a vector of all zeros in place of the labels that the model has been trained on
    if Y is not None:
        Y_zeros = np.zeros((n_items, network_architecture['n_labels'])).astype('float32')
    else:
        Y_zeros = None
    
    # create np arrays to store regularization strengths, which we'll update outside of the tensorflow model
    if regularize:
        l2_strengths = 0.5 * np.ones([n_topics, dv]) / float(n_train)
        l2_strengths_c = 0.5 * np.ones([model.beta_c_length, dv]) / float(n_train)
        l2_strengths_ci = 0.5 * np.ones([model.beta_ci_length, dv]) / float(n_train)
    else:
        l2_strengths = np.zeros([n_topics, dv])
        l2_strengths_c = np.zeros([model.beta_c_length, dv])
        l2_strengths_ci = np.zeros([model.beta_ci_length, dv])

    total_batch = int(n_items / batch_size)
    display_step = int(total_batch*0.2)

    eta_bn_prop = 0.0  # interpolation between batch norm and no batch norm in final layer of recon
    kld_weight = 1.0  # could use this to anneal KLD, but not currently doing so

    # Start evaluation cycle
    avg_loss = 0.
    avg_task_loss = 0.
    task_loss = 0.
    if task == "class":
        task_name = "accuracy"
    elif task == "reg":
        task_name = "mse"
    # Loop over all observations
    if Y is not None:
        for i in range(total_batch):
            # get a minibatch
            ixs = range(batch_size*i,batch_size*(i+1))
            if Y_zeros is not None and C is not None:
                obs_xs, obs_ys, obs_y_zeros, obs_cs = X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), Y_zeros[ixs, :].astype('float32'), C[ixs, :].astype('float32')
            elif Y_zeros is not None:
                obs_xs, obs_ys, obs_y_zeros, obs_cs = X[ixs, :].astype('float32'), Y[ixs, :].astype('float32'), Y_zeros[ixs, :].astype('float32'), None
            elif C is not None:
                obs_xs, obs_ys, obs_y_zeros, obs_cs =  X[ixs, :].astype('float32'), None, None, C[ixs, :].astype('float32')
            else:
                obs_xs, obs_ys, obs_y_zeros, obs_cs = X[ixs, :].astype('float32'), None, None, None
            # do one update, passing in the data, regularization strengths, and bn
            loss, task_loss_i, pred, theta = model.predict(X=obs_xs, Y=obs_y_zeros, C=obs_cs, l2_strengths=l2_strengths, 
                                             l2_strengths_c=l2_strengths_c, l2_strengths_ci=l2_strengths_ci,
                                             eta_bn_prop=eta_bn_prop, kld_weight=kld_weight,is_training=False)
            # compute accuracy/mse on prediction
            if network_architecture['n_labels'] > 0 and task == "class":
                task_loss += np.sum(pred == np.argmax(obs_ys, axis=1)) / float(n_items) # accuracy
            elif network_architecture['n_labels'] > 0 and task == "reg":
                task_loss += np.sum((obs_ys - pred)**2) / float(n_items) # mse
            # Compute average loss
            avg_loss += loss / n_items
            avg_task_loss += task_loss_i / n_items
            # Save predictions
            if i==0:
                predictions = pred
                thetas = theta
            else:
                predictions = np.append(predictions,pred,axis=0)
                thetas = np.append(thetas,theta,axis=0)

            if np.isnan(avg_loss):
                print(i, np.sum(obs_xs, 1).astype(np.int), obs_xs.shape)
                print('Encountered NaN, stopping evaluation. Please check the learning_rate settings and the momentum.')
                # return vae,emb
                sys.exit()

            # Display logs per observations step
            if verbose_updates:
                if i % display_step == 0:
                    if network_architecture['n_labels'] > 0:
                        print("EVAL {} | ".format(subset),"until obs:", '%d' % i, "; loss =", "{:.9f}".format(avg_loss),
                              "; avg_{}_loss =".format(task_name), "{:.9f}".format(avg_task_loss), 
                              "; eval {} so far (noisy) =".format(task_name), "{:.9f}".format(task_loss))
                        if batch_size > 1:
                            print("EVAL {} | y_act {}:\n".format(subset,i), obs_ys[:3])
                            print("EVAL {} | y_pred {}:\n".format(subset,i), pred[:3])
                        else:
                            print("EVAL {} | y_act {}:\n".format(subset,i), obs_ys)
                            print("EVAL {} | y_pred {}:\n".format(subset,i), pred)
                    else:
                        print("EVAL {} | ", "Obs:".format(subset), '%d' % i, "loss=", "{:.9f}".format(avg_loss))
    else:
        # Thetas
        thetas = model.compute_theta(X, Y, C)
        avg_loss = np.mean(model.get_losses(X, Y, C))
        predictions = np.nan
        task_loss = np.nan
        avg_task_loss = np.nan
                     
    # Eval perplexity
    eval_perplexity_all, eval_perplexity_text = evaluate_perplexity(model, X, Y, C)
    
    # Save
    if save_results and output_dir is not None:
        print("{} | perplexity = {:.4f}".format(subset, eval_perplexity_text))
        print("{} | {}:".format(subset, task_name),task_loss)
        print("{} | avg. loss:".format(subset),avg_loss)
        
        # Thetas
        np.savez(os.path.join(output_dir, 'theta.'+subset+'.npz'), theta=thetas)
        
        # save perplexities
        pd.Series(data= eval_perplexity_all, name = "perplexity").to_csv(os.path.join(output_dir,'perplexity_all.'+subset+'.csv'))
        pd.Series(data= eval_perplexity_text, name = "perplexity").to_csv(os.path.join(output_dir,'perplexity_text.'+subset+'.csv'))
        
        if Y is not None:
            # Actuals and predictions
            pred_series = pd.DataFrame(predictions)
            y_series = pd.DataFrame(Y)
            y_series.to_csv(os.path.join(output_dir,'y_actuals.'+subset+'.csv'))
            pred_series.to_csv(os.path.join(output_dir,'y_predictions.'+subset+'.csv'))

            if task == "reg":
                # overall
                pR = 1-(task_loss/np.var(Y))
                print("{} | R^2 on labels = {:.4f}".format(subset, pR))
                print("{} | variance of y:".format(subset), np.var(Y))
                pd.Series(data = task_loss, name = "mse").to_csv(os.path.join(output_dir,"mse."+subset+'.csv'))
                pd.Series(data= pR, name = "pR2").to_csv(os.path.join(output_dir,"pR2."+subset+'.csv'))  
                # separate targets
                if pred_series.shape[1] >1:
                    target_mses = np.sum((y_series - pred_series)**2) / float(y_series.shape[0])
                    target_pRs = 1-(target_mses/np.var(y_series))
                    print("{} | target-variance of y:".format(subset), np.var(y_series).values)
                    print("{} | target-mse on labels =".format(subset), target_mses.values)
                    print("{} | target-R^2 on labels = ".format(subset), target_pRs.values)
                    pd.Series(data = task_loss, name = "mse").to_csv(os.path.join(output_dir,"mse."+subset+'.csv'))
                    pd.Series(data= pR, name = "pR2").to_csv(os.path.join(output_dir,"pR2."+subset+'.csv'))  
                    if pred_series.shape[1] >1:
                        target_mses.to_csv(os.path.join(output_dir,'mses_per_target.'+subset+'.csv'))
                        target_pRs.to_csv(os.path.join(output_dir,'pRs_per_target.'+subset+'.csv'))  

    
    return predictions, task_loss, avg_loss, avg_task_loss, eval_perplexity_all, eval_perplexity_text
    


    
    
if __name__ == '__main__':
    main()
