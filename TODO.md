# Tasks

(Descending Priority)

- [ ] Ensure that all the parts of tehe computation graph are connecting and calcuating things.
    - For this stuff I can imagine use using stuff like `torchviz` to visualizing. I tried but `torchviz` reveals too much.\
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
