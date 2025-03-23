from torch import Dataset
import json

from multihopkg.logging import setup_logger
from multihopkg.run_configs.pretraining import get_args
from multihopkg.knowledge_graph import SunKnowledgeGraph
from multihopkg import data_utils


def processe_dataset(location: str) -> torch.utils.data.Dataset:
    columns = []
    with open(location, "r") as f:
        json_file = json.load(f)

    columns = ["questions", "answer", "triples", "triples_labeled"]
    rows = []
    for sample in json_file:
        questions = sample["questions"]
        encoded_questions = data_utils.encode_questions(questions, answer_tokenizer)

        rows.append[
            samples["questions"],
            samples["answer"],
            samples["orig"]["triples"],
            samples["orig"]["triples_labeled"],
        ]

    
    dataset = Dataset(columns, rows)

    return dataset

def main():
    args = get_args()

    # Lets process the dataset
    dataset = processe_dataset(args.data_dir)

    # Load the pretrained embeddings

    # TODO: Dataloader somewhere around here


    # Load the dictionaries
    id2ent, ent2id, id2rel, rel2id = data_utils.load_dictionaries(args.data_dir)

    # Load the sun model
    knowledge_graph = SunKnowledgeGraph(
        model=args.model,
        pretrained_sun_model_path=args.pretrained_sun_model_loc,
        data_path=args.data_dir,
        graph_embed_model_name=args.graph_embed_model_name,
        gamma=args.gamma,
        id2entity=id2ent,
        entity2id=ent2id,
        id2relation=id2rel,
        relation2id=rel2id,
        device=args.device,
    )

    # We need to Load the Bart Model.
    # We prepare our custom encoder for Bart Here
    hunch_llm = GraphBart(
        pretrained_bart_model=args.pretrained_llm_for_hunch,
        answer_tokenizer=answer_tokenizer,
        # We convert the graph embeddings to state embeddings obeying current state dimensions
        graph_embedding_dim=args.llm_model_dim,
    ).to(args.device)




if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
   main()
