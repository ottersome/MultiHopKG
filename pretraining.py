from multihopkg.logging import setup_logger
from multihopkg.run_configs.pretraining import get_args
from multihopkg.knowledge_graph import SunKnowledgeGraph
from multihopkg import data_utils

def main():
    args = get_args()

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



if __name__ == "__main__":
    logger = setup_logger("__PRETRAINING_MAIN__")
    main()
