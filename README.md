# Aspect Based Sentiment Analysis on Laptop Reviews
The goal of this project is to identify opinions expressed within laptop reviews in terms of its attributes using a granular technique known as Aspect Based Sentiment Analysis. 

## Problem Definition
There are two components that we are interested in identifying:
1. Aspect: composed of either an attribute or a subcomponent of the entity (laptop).
2. Sentiment: each opinion is charged with a sentiment that can be either positive or negative.

## Aspects Detection and Extraction
(For a list of all aspects, see below)


Aspect detection can be interpreted as a multi-class classification problem. The goal is to identify the aspects that are alluded in each review. Given the size of the tag set, we implemented a binary Logistic Regression classifier for each aspect that takes as input features the bag-of-words (BOW) of each sentence and its tags. The solver used was “newton-cg”, which supports multinomial loss.


Aspect extraction consists in identifying the tokens in the text that allude to these categories. These will eventually help us find the opinions associated to each aspect. For this, the frequent noun phrase approach proposed by Hu and Liu (2004) was implemented, enhanced by the application of Positive Pointwise Mutual Information (PPMI).

## Polarity Detection
This subtask actually consists of two steps: 
1. Identifying the opinions regarding the features found. For this, we will use the keyword opinion list provided by Hu and Liu.
2. Determining the polarity of these opinions. For this, we will implement the lexicon- based approach proposed by Ding, Liu and Yu (2008). 

## List of Aspects
- BATTERY
- COMPANY
- CPU
- DISPLAY
- FANS_COOLING
- GRAPHICS
- HARD_DISC
- HARDWARE
- KEYBOARD
- MEMORY
- MOTHERBOARD
- MOUSE
- MULTIMEDIA_DEVICES
- OPTICAL_DRIVES 
- OS
- PORTS 
- POWER_SUPPLY 
- SHIPPING
- SOFTWARE 
- SUPPORT
- WARRANTY
- CONNECTIVITY
- DESIGN_FEATURES
- GENERAL
- MISCELLANEOUS
- OPERATION_ PERFORMANCE
- PORTABILITY 
- PRICE 
- QUALITY 
- USABILITY
