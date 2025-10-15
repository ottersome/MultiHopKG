import torch
import debugpy, platform


STATES = torch.zeros((6,3,10))
WRITE_PTR = torch.zeros(6, dtype=torch.long)

def add_question_experiences(
    question_n_exp_idxs: torch.Tensor,
    cur_states: torch.Tensor,            # [E, A]
):
    """
    Performs a round-robin write into the replay buffer for multiple questions at once.
    """
    E, A  = cur_states.shape
    cap = 3 

    qids, qids_count = torch.unique(question_n_exp_idxs, return_counts=True)

    # current write positions for each qid
    start = WRITE_PTR[question_n_exp_idxs] # [E]
    arange_N = torch.concat([ torch.arange(0,qid_count.item()) for  qid_count in qids_count ]) % cap
    idxs = (start[:, None] + arange_N[None, :]) % cap  # [B, N]

    # Write into buffer (parallelized)
    STATES[question_n_exp_idxs, arange_N, :] = cur_states
    # Advance write pointer
    WRITE_PTR[qids] = (start + qids_count) % cap

def __len__(self) -> int:
    return self._total_size

def main():
    debugpy.listen(("0.0.0.0", 42023))
    print("debugpy listening on :0.0.0.0, 42023", flush=True)
    debugpy.wait_for_client()

    question_n_exp_idxs = torch.LongTensor(
        [1,1,2,2,3,4,5]
    ) 
    cur_states = torch.rand((7,10))
    add_question_experiences(question_n_exp_idxs, cur_states)


if __name__ == "__main__":
    main()
