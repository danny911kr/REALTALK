# REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2502.13270-b31b1b.svg)](https://arxiv.org/abs/2502.13270)

This repo provides the model, code & data of our paper: "REALTALK: A 21-Day Real-World Dataset for Long-Term Conversation".

## Overview
Long-term, open-domain dialogue capabilities are essential for chatbots aiming to recall past interactions and demonstrate emotional intelligence (EI). Yet, most existing research relies on synthetic, LLM-generated data, leaving open questions about real-world conversational patterns. To address this gap, we introduce REALTALK, a 21-day corpus of authentic messaging app dialogues, providing a direct benchmark against genuine human interactions.
We first conduct a dataset analysis, focusing on EI attributes and persona consistency to understand the unique challenges posed by real-world dialogues. By comparing with LLM-generated conversations, we highlight key differences, including diverse emotional expressions and variations in persona stability that synthetic dialogues often fail to capture.

Building on these insights, we introduce two benchmark tasks: (1) persona simulation where a model continues a conversation on behalf of a specific user given prior dialogue context; and (2) memory probing where a model answers targeted questions requiring long-term memory of past interactions.
Our findings reveal that models struggle to simulate a user solely from dialogue history, while fine-tuning on specific user chats improves persona emulation. Additionally, existing models face significant challenges in recalling and leveraging long-term context within real-world conversations.
## Table of contents

1. [Setup](#setup) [Coming soon]
2. [Data](#data)
3. [Task](#task) [Coming soon]


<hr/>

## Data

- `data/*.json`: processed data of REALTALK dataset. (json format)
- `data/raw`: Contains the raw data of REALTALK dataset. (xlsx format)
