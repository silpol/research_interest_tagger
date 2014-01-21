research_interest_tagger
========================

Guesses applicable research interests for text documents

## Dependencies

Python 2.7, Scikit-Learn, Twisted

## Sample setup & usage

On Ubuntu/Debian:

```
  # Install dependencies
  sudo apt-get update
  sudo apt-get install -y build-essential libatlas-dev liblapack-dev gfortran python-dev python-pip

  sudo pip install numpy --upgrade
  sudo pip install scipy --upgrade
  sudo pip install scikit-learn
  sudo pip install twisted --upgrade

  # Download source and models
  git clone git@github.com:academia-edu/research_interest_tagger.git
  cd research_interest_tagger

  wget https://s3.amazonaws.com/work_tagging/models-1386299128.tgz # Trained on Academia.edu data--please respect our data transfer charges and don't download unnecessarily
  mkdir models_dir
  tar -xzf models-1386299128.tgz -C models_dir

  # Start server
  nohup python classification_server.py models_dir &

  # After a few seconds, this should return "{ health: ok }"
  curl http://localhost:6606/health

  # Download a test file--"Computing Machinery and Intelligence" by Alan Turing--and convert to text
  sudo apt-get install poppler-utils # Not a dependency, just useful for text extraction from PDFs
  wget http://orium.pw/paper/turingai.pdf
  pdftotext turingai.pdf

  # Try out the classifier
  curl http://localhost:6606/classify -F text=@turingai.txt
```

## Copying

Copyright Academia.edu.

MIT licensed--see LICENSE.txt.

Many thanks to the authors of scikit-learn, who did all the hard work.
