parliasent.py

Speech-level sentiment analysis for Hansard UK parliamentary debate transcripts

For use with the “HanDeSeT: Hansard Debates with Sentiment Tags” corpus (Abercrombie, G. & Batista-Navarro, R., 2018), available at Mendeley Data https://data.mendeley.com/datasets/xsvp45cbt4/

See "'Aye' or 'No'? Speech-level Sentiment Analysis \\of Hansard UK Parliamentary Debate Transcripts" (Abercrombie, G. & Batista-Navarro, N., 2018) for further details.


Sentiment analysis can be run using any combination of the following models, sentiment labels, and features:

  Models:
  1  = one-step "Speech" model.
  2a = two-step "Motion-speech" model with automatic motion classification.
  2b = two-step "Motion speech" model with Goverment/opposition motion labelling.

  Sentiment polarity labels:
  vote = division vote labels.
  manual = manually annotated labels.
  
  Features:
  0 = Textual features only
  1 = Textual features + speaker party affiliation
  2 = Textual features + speaker party affiliation + debate ID
  3 = Textual features + speaker party affiliation + debate ID + motion party affiliation
  4 = Speaker party affiliation + debate ID
  5 = Speaker party affiliation + debate ID + motion party affiliation


To run sentiment analysis:

1) Ensure that `parliasent.py` and `HanDeSeT.csv` are located in the same folder.
2) Run:
  `python` `parliasent.py` `<model>` `<sentiment_label features>`
  
E.g., to run the two-step model with automatic motion classification, manually applied sentiment labels, and textual features, speaker party affiliation, and debate ID, enter: 
  
  python parliasent.py 2a manual 3
  
