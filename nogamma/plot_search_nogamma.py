import numpy as np
from sklearn.cluster import KMeans
from behavenet.plotting.cond_ae_utils import (
    plot_psvae_training_curves,
    plot_label_reconstructions,
    plot_hyperparameter_search_results,
    make_latent_traversal_movie
)
from nogamma.hyperparameter_search_nogamma import (
    hyperparameter_search
)
from nogamma.psvae_experiment_nogamma import PSvaeExperiment
from nogamma.search_utils_nogamma import (
    load_latents_trials_frames,
    list_hparams
)


def _cluster(latents, n_clusters=5):
    # np.random.seed(0)  # to reproduce clusters
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    distances = kmeans.fit_transform(latents)
    clust_id = kmeans.predict(latents)
    return distances, clust_id


def plot_psvae_training_curves_wrapper(lab, expt, animal, session, expt_name, n_ae_latents, rng_seeds_model, n_labels,
                                       which_as_array='alphas', alpha=None, beta=None, save_file=None, **kwargs):
    alphas, betas = list_hparams(lab, expt, animal, session, expt_name, n_ae_latents)
    n_ae_latents = [n_ae_latents - n_labels]
    if alpha is None:
        alpha = alphas[0]
    if beta is None:
        beta = betas[0]
    if save_file is None:
        save_file = 'psvae_training_alpha-{}_beta-{}'.format(alpha, beta)
    if which_as_array == 'alphas':
        try:
            betas = [beta]
            rng_seeds_model = rng_seeds_model[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'betas':
        try:
            alphas = [alpha]
            rng_seeds_model = rng_seeds_model[:1]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'rng_seeds_model':
        try:
            betas = [beta]
            alphas = [alpha]
            n_ae_latents = n_ae_latents[:1]
        except TypeError:
            pass
    if which_as_array == 'n_ae_latents':
        try:
            betas = [beta]
            rng_seeds_model = rng_seeds_model[:1]
            alphas = [alpha]
        except TypeError:
            pass

    plot_psvae_training_curves(lab, expt, animal, session, alphas, betas, n_ae_latents, rng_seeds_model,
                               expt_name, n_labels, save_file=save_file, **kwargs)


def plot_label_reconstructions_wrapper(lab, expt, animal, session, n_ae_latents, experiment_name, n_labels, trials,
                                       alpha, beta, save_file=None, **kwargs):
    if save_file is None:
        save_file = 'label_reconstruct_trials-{}_alpha-{}_beta-{}'.format(trials, alpha, beta)
    n_ae_latents -= n_labels
    plot_label_reconstructions(lab, expt, animal, session, n_ae_latents, experiment_name,
                               n_labels, trials, alpha=alpha, beta=beta, save_file=save_file, **kwargs)


def make_latent_traversal_movie_wrapper(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                                        model_class='ps-vae', n_clusters=10, rng_seed_model=0, save_file=None,
                                        **kwargs):
    if save_file is None:
        save_file = 'latent_movie_alpha-{}_beta-{}'.format(alpha, beta)

    movie_expt = PSvaeExperiment(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                                 model_class=model_class, **kwargs)

    # need to test model_ae in load_latents_trials_frames
    latents, trials, frames = load_latents_trials_frames(movie_expt.hparams, movie_expt.version)
    traversal_trials = []
    batch_idxs = []
    distances, _ = _cluster(latents, n_clusters)
    for clust in range(n_clusters):
        frame_idx = np.argmin(distances[:, clust])
        traversal_trials.append(trials[frame_idx])
        batch_idxs.append(frames[frame_idx])
    trial_idxs = [None] * len(traversal_trials)
    make_latent_traversal_movie(lab, expt, animal, session, model_class, alpha, beta,
                                n_ae_latents - len(label_names), rng_seed_model, expt_name, len(label_names),
                                trial_idxs, batch_idxs, traversal_trials, save_file=save_file, **kwargs)


def plot_hyperparameter_search_results_wrapper(lab, expt, animal, session, n_labels, label_names, n_ae_latents,
                                               expt_name, alpha, beta, save_file=None, beta_n_ae_latents=None,
                                               beta_expt_name=None, batch_size=None, format='pdf', **kwargs):
    if save_file is None:
        save_file = 'hparam_search_results_alpha-{}_beta-{}'.format(alpha, beta)
    if beta_n_ae_latents is None:
        beta_n_ae_latents = n_ae_latents - n_labels
    if beta_expt_name is None:
        beta_expt_name = expt_name
    alpha_weights, beta_weights = list_hparams(lab, expt, animal, session, expt_name, n_ae_latents)
    alpha_n_ae_latents = [n_ae_latents - n_labels]
    plot_hyperparameter_search_results(lab, expt, animal, session, n_labels, label_names, alpha_weights,
                                       alpha_n_ae_latents, expt_name, beta_weights,
                                       beta_n_ae_latents, beta_expt_name, alpha, beta, save_file,
                                       batch_size=batch_size, format=format, **kwargs)


def plot_and_film_best(lab, expt, animal, session, label_names, expt_name, n_ae_latents, trials, beta=1,
                       model_class='ps-vae', n_clusters=5, rng_seed_model=0, rng_seeds_model=None, save_file=None,
                       **kwargs):
    if rng_seeds_model is None:
        rng_seeds_model = [rng_seed_model]
    alphas, betas = list_hparams(lab, expt, animal, session, expt_name, n_ae_latents)
    alpha, beta = hyperparameter_search(lab, expt, animal, session, label_names, expt_name, n_ae_latents,
                                        alphas, betas, beta=beta, **kwargs)
    print("Using alpha: {} and beta: {} from hyperparameter search".format(alpha, beta))

    # do for every model, save in version folder
    for setting in ['alphas', 'betas', 'rng_seeds_model', 'n_ae_latents']:
        pass

    # save hparam plots in expt_name directory, one for each alpha
    for alpha_ in alphas:
        plot_hyperparameter_search_results_wrapper(lab, expt, animal, session, len(label_names), label_names,
                                                   n_ae_latents, expt_name, alpha_, beta, save_file)

    # one setting alpha to best and search over beta
    for beta_ in betas:
        plot_hyperparameter_search_results_wrapper(lab, expt, animal, session, len(label_names), label_names,
                                                   n_ae_latents, expt_name, alpha, beta_, save_file)

    plot_psvae_training_curves_wrapper(lab, expt, animal, session, expt_name, n_ae_latents, rng_seeds_model,
                                       len(label_names), **kwargs)
    plot_label_reconstructions_wrapper(lab, expt, animal, session, n_ae_latents, expt_name, len(label_names), trials,
                                       alpha, beta, save_file, **kwargs)
    make_latent_traversal_movie_wrapper(lab, expt, animal, session, label_names, expt_name, n_ae_latents, alpha, beta,
                                        model_class=model_class, n_clusters=n_clusters,
                                        rng_seed_model=rng_seed_model, save_file=save_file, **kwargs)
