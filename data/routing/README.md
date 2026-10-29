# Banking77 support questions

`train.csv`, `test.csv` and `categories.json` are the Banking77 dataset:
13,083 real customer-support questions from online banking, each labelled
with one of 77 intents (10,003 train, 3,080 test, 40 test questions per
intent).

- Source: PolyAI, <https://github.com/PolyAI-LDN/task-specific-datasets>
- Paper: Casanueva et al., "Efficient Intent Detection with Dual Sentence
  Encoders", NLP4ConvAI workshop, ACL 2020
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0). Redistributed
  here unchanged except for line endings (CRLF converted to LF).

This is a different domain from the StreamFlix help-centre corpus used in
notebooks 01–10. It is used in notebook 11 because routing needs labelled
tickets in volume, and inventing 13,000 synthetic ones would test the
generator rather than the router. See notebook 11, section A.
