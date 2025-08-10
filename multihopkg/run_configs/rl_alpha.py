"""
Instead of using bach config filees, we will just separate them into different files we can just import

This is config `alpha.py`:
Just to try to get the first run running.
"""

import argparse

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # General Training Settings
    ap.add_argument('--device', type=str, default="cuda:0", help="Device to run the model on (e.g., cuda:0 or cpu)")
    ap.add_argument("--seed", type=int, default=420, metavar="S", help="Random seed for reproducibility")
    ap.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with extra logs") # TODO: Check if this is still needed.
    ap.add_argument('--verbose','-v',action="store_true", help="Print, log, and dump evaluation results")
    ap.add_argument('--visualize','-vv',action="store_true", help="Render 2D/3D visualizations of evaluation results in addition to printing and logging")
    ap.add_argument('--track_gradients', '-g', action='store_true', help='Track and log gradients during training')
    ap.add_argument('--preferred_config', type=str, default="configs/rl_config.yaml", help="Path to YAML configuration file (default: configs/my_config.yaml). " \
                        "If not empty, overrides the respective command line arguments.")

    # Learning Hyperparameters
    ap.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for optimizer (default: 1e-5)')
    ap.add_argument('--beta', type=float, default=0.0, help='Entropy regularization coefficient (default: 0.0)') # TODO: Check if this is still used.
    ap.add_argument('--gamma', type=float, default=12, help='Margin or scaling factor used by the knowledge graph embedding model.' \
                            'Not actually used during flexible translation (default: 12)') # TODO: Force load this value from the state_dict instead. 
    ap.add_argument('--rl_gamma', type=float, default=0.9, help='Discount factor for the reinforcement learning agent (default: 0.9)')
    ap.add_argument('--baseline', type=str, default='n/a', help='Baseline strategy for policy gradient (default: n/a)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    
    ap.add_argument('--supervised_adapter_scalar', default=0.5, type=float, help='Scalar for the supervised adapter loss (default: 0.0)') 
    ap.add_argument('--supervised_sigma_scalar', default=0.1, type=float, help='Scalar for the supervised sigma loss (default: 0.1)')
    ap.add_argument('--supervised_expected_sigma', default=0.1, type=float, help='Scalar for the supervised expected sigma value (default: 0.1)') 
                    #TODO: Add the warmup parameters here


    # Dropout Scheduling
    ap.add_argument('--action_dropout_rate', type=float, default=0.1, help='Dropout rate for randomly masking out knowledge graph edges (default: 0.1)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    ap.add_argument('--action_dropout_anneal_factor', type=float, default=0.95, help='Decrease the action dropout rate once the dev set results stopped increase (default: 0.95)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    ap.add_argument('--action_dropout_anneal_interval', type=int, default=1000, help='Number of epochs to wait before decreasing the action dropout rate (default: 1000. Action '
                         'dropout annealing is not used when the value is >= 1000.)') # TODO: Check if this is still used. Answer: Passed to NavigationAgent, but not used in the current implementation.
    
    # Training Duration & Rollout
    ap.add_argument('--epochs',type=int,default=200, help='Total number of training epochs (default: 200)')
    ap.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training instead of just observing (default: 0)') # TODO: Implement this.
    ap.add_argument('--num_rollouts', type=int, default=5, help='Number simultaneous trajectories to sample from during training for each sample question (default: 1)')
    ap.add_argument('--num_rollout_steps', type=int, default=8, help='Maximum number of steps per questions (default: 8)')

    # Batch Settings
    ap.add_argument('--batch_size', type=int, default=256, help='Training mini-batch size (default: 256)')
    ap.add_argument('--batch_size_dev', type=int, default=64, help='Evaluation mini-batch size (default: 64)')
    ap.add_argument('--batches_b4_eval', type=int, default=100, help='Batches to train before first evaluation phase (default: 100)') #TODO: Remove if unused.
    ap.add_argument('--num_batches_till_eval', type=int, default=15, help='Batches to train between evaluations (default: 15)')

    # Datasets & File Paths
    # QA Dataset
    ap.add_argument('--raw_QAData_path', type=str, default="./data/mquake/mquake_qna_ds.csv", help="Path to the raw QA CSV dataset (default: FreebaseQA)")
    ap.add_argument('--cached_QAMetaData_path', type=str, default="./.cache/mquake/mquake_clean.json", help="Path to cached tokenized QA metadata JSON file")
    ap.add_argument('--force_data_prepro', '-f', action="store_true", help="Force re-processing of QA data, even if cache exists")
    ap.add_argument('--use_kge_question_embedding', '-kq', action="store_true", help="Use entity and relation embedding as the questions instead of textual question. Only valid for single-hop task. (default: False)")
    #TODO: Add the override_split option here

    # GTLLm Parameters
    ap.add_argument('--pretrained_gtllm_path', default="models/gtllm/flagship.pt", type=str)
    ap.add_argument('--frozen_llm_weights', action="store_true", default="If true, it will freeze HunchBart llm weights")
    
    # Navigation Agent Settings
    ap.add_argument('--nav_start_emb_type', type=str, default="centroid", help="Initial navigation point: 'centroid', 'random', or 'relevant'")
    ap.add_argument('--nav_epsilon_error', type=float, default=50.0, help="Allowable distance to consider the answer as 'reached' (default: 50.0)")
    ap.add_argument('--nav_epsilon_metric', type=str, default="l2", help="Distance metric for navigation: 'l1', 'l2', or 'deg' (default: l2)")
    ap.add_argument('-ts', '--add_transition_state', action='store_true', help="Include the past position, action, and current position into the state (default: False)")

    # RNN Settings
    ap.add_argument('--history_dim', type=int, default=768, metavar='H', help='Hidden size of action history LSTM encoder (default: 768)')
    ap.add_argument('--history_num_layers', type=int, default=3, metavar='L', help='Number of layers in action history LSTM encoder (default: 3)')
    ap.add_argument('--ff_dropout_rate', type=float, default=0.1, help='Dropout rate for feed-forward layers (default: 0.1)')
    ap.add_argument('--rnn_hidden',type=int,default=400, help='Hidden size of general-purpose RNN modules (default: 400)')

    # Textual Embedding (LLMs)
	# TODO: (eventually) We might want to add option of locally trained models.
    # TODO: Replace the redundant models here (question_tokenizer_name vs question_embedding_model) and (answer_tokenizer_name vs pretrained_llm_for_hunch)
    

    'Logging and Experiment Tracking'
    ap.add_argument("-w", "--wandb", action="store_true", help="Enable Weights & Biases experiment tracking")
    ap.add_argument("--wandb_project_name", type=str, help="wandb: Project name to group runs")
    ap.add_argument("--wr_name", type=str, help="wandb: Unique name for this run")
    ap.add_argument("--wr_notes", type=str, help="wandb: Additional notes for this run")

    return ap.parse_args()

