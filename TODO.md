# Tasks

(Descending Priority)

- [ ] Try to retrain the embedding models
- [ ] Suspicion that the continuous action space is not learning properly
- [ ] Ensure that all the parts of tehe computation graph are connecting and calcuating things.
    - For this stuff I can imagine use using stuff like `torchviz` to visualizing. I tried but `torchviz` reveals too much.\
- [ ] Writing a suite of tools for  troubleshooting.
- [ ] Dump the angle/distance/metric taken at every step during path traversal.
- [ ] Check that the Continuous action space is sampled probperly, and mu and sigma are learned, ow, maybe agent gets stuck
- [ ] Check for vanishing gradient
    - Use the heatmap (layerwise) to see if model learns
    - Store the parameters of a fitted distribution on a per layer and epoch basis to inspect vanishing gradients.
    - Other techniques also apply
- [x] Create evaluation to ensure that the language model is dumping reasonable answers.
- [x] Create evaluation to show the path taken for an answer. 
- [ ] Create a benchmark for evaluation
    - Lots of thinking required here.
    - Option 1: Use the original Salesforce MultiHopKG with FB15k and FreebaseQA (triplets only)
- [x] Create script for splitting tripplet dataset into different splits
- [x] Implement Knowledge base well enough to the point that we can change it to our will.
    - [x] Find how to generate the `entity2idx.txt` in our implementation of the thing.

# Later
- [ ] Understand where beamsearch should be placed
- [ ] Change the teachering force to be operating on the graph nodes and predict them, instead of words
