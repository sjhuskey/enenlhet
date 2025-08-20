---
title: "Observations on Fine-Tuning Whisper for ASR of Enenlhet"
author: "Samuel J. Huskey"
date: 2025-07-10
---

Part of the summer AI research grant that I received from OU is devoted to helping my colleague Raina Heaton with an Audio Speech Recognition model fine-tuned for Enenlhet, a low-resource language spoken by a couple thousand people in Paraguay.

Building on the work that Heaton did previously with two researchers at Boston University ([Enenlhet as a case-study to investigate ASR model generalizability for language documentation](https://aclanthology.org/2024.americasnlp-1.15/) (Le Ferrand et al., AmericasNLP 2024)), I have been working on fine-tuning a Whisper model. Like the previous researchers, I am using the tutorial "[Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper#prepare-environment)" by Sanchit Gandhi as a guide.

Since I have encountered obstacles at practically every step, I thought it would be best to summarize them here along with the solutions and/or work-arounds that I have had to implement. 

The obstacles stem from three main issues. First, Enenlhet does not have a nice, clean, pre-formatted dataset like the one Gandhi uses for Hindhi, which he downloaded from [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) for the tutorial. Second, as nice as Hugging Face's API is, its documentation is sometimes sparse and difficult to follow, and some of its components are not always in synch with each other. The third issue is that ASR models require large storage capacity and fast processors. Even then, the work takes a long time from start to finish.

## The Dataset

### Managing the data

The first problem I encountered was simply obtaining the raw data. The zip archive of the files is over 20 GB, which makes it difficult to share. OU recently restricted the use of detachable hard drives, so Raina had to share the files with me on SharePoint. I encountered difficulties extracting the files from the zip archive because some of the information was corrupt. I was able to repair the damage and extract the files.

Because the raw data is 20 GB, I can't easily transfer it to Google Colab, where I could speed up the process of building the dataset by using a GPU. Although I have a MacBook Pro with an M4 GPU, I can't use its capabilities because it is not compatible with training Whisper models (as I learned from a couple of days trying to get it to work!). Therefore, I had to resort to building the dataset on my laptop's CPU, which took about five hours each time. I also ran into storage problems, including once in the middle of the night, when I checked the process and found that my laptop's 500 GB hard drive had maxed out! Of course, the dataset wasn't the only thing on the laptop, but that episode highlights a significant problem with working on a laptop.

### The `prepare_dataset()` function

The raw data consists of `.wav` files and `.eaf` ("ELAN") files. That information must be converted into a format that the model can use. I knew already that Whisper models require all audio data to use the same sampling size (16,000 kHz), so I first tried to resample and segment the files manually. I also manually parsed the ELAN files and associated them with their respective segments. Unfortunately, that dataset turned out to be highly flawed, as demonstrated by the result of the first attempt to fine-tune the model.

I tried again, this time using Hugging Face's `datatset` class to handle the sampling and segmentation. That went better. The bonus was that I could automate the task to uploading the dataset to the Hugging Face Hub. The downside was that the dataset was still not quite in the right format because I had not included `return_attention_mask=True` in the `prepare_dataset()` function. Gandhi says nothing about this, probably because his data is more uniform and of higher quality than Heaton's field data. His `prepare_dataset()` function looks like this:

```python
def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch
```

Mine looks like this:

```python
def prepare_dataset(batch):
    # Extract Whisper-compatible inputs (features + attention mask)
    inputs = processor(
        batch["audio"]["array"],
        sampling_rate=batch["audio"]["sampling_rate"],
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Tokenize the text labels (no context manager needed)
    labels = processor.tokenizer(
        batch["text"],
        return_tensors="pt",
        padding="longest",
        truncation=True,  # Optional but safe
    )

    batch["input_features"] = inputs.input_features[0]
    batch["attention_mask"] = inputs.attention_mask[0]
    batch["labels"] = labels.input_ids[0]
    return batch
```

Both do similar things, but the `attention_mask` standardizes the length of the inputs so that they do not vary widely from one to another. That's necessary with the kind of data I am working with.

### The `tokenizer`

The tokenizer is another departure from Gandhi's tutorial. A tokenizer processes text and turns it into, well, "tokens." Put simply, tokens are the individual components (e.g., words, punctuation, etc.) of a text. Many languages (like Hindhi) already have tokenizers from previous research and applications. If there isn't a tailor-made tokenizer, you have to fall back to a generic, default tokenizer that does not capture the unique features of a given language. So, where Gandhi can do this:

```python
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Hindi", task="transcribe")
```

I am stuck doing this:

```python
# Load processor, which includes both the feature extractor and tokenizer
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer
```

That gives me a generic tokenizer, which is okay, but it fails to capture the nuances unique to Enenlhet.

### Uploading the dataset

Originally, I tried using Git to upload the dataset to the repository I created on Hugging Face Hub. Owing to the size of the files, I installed [Git Large File Storage](https://git-lfs.com/) and set it up to track the large files. For example, I tracked the `.wav` files like this:

```bash
git rm --cached  '*.wav'
git lfs track '*.wav'
git add '*.wav'
git commit -m "Track large wav files with Git LFS"
git push
```

But I encountered error after error with this method of uploading the files. For example, several times in a row the process failed and generated this error:

```bash
Uploading LFS objects:  64% (9/14), 6.1 GB | 1.4 MB/s, done.                    Authentication required: Password authentication in git is no longer supported. You must use a user access token or an SSH key instead. See https://huggingface.co/blog/password-git-deprecation
Authentication required: Password authentication in git is no longer supported. You must use a user access token or an SSH key instead. See https://huggingface.co/blog/password-git-deprecation
Authentication required: Password authentication in git is no longer supported. You must use a user access token or an SSH key instead. See https://huggingface.co/blog/password-git-deprecation
Authentication required: Password authentication in git is no longer supported. You must use a user access token or an SSH key instead. See https://huggingface.co/blog/password-git-deprecation
Authentication required: Password authentication in git is no longer supported. You must use a user access token or an SSH key instead. See https://huggingface.co/blog/password-git-deprecation
error: failed to push some refs to 'hf.co:datasets/sjhuskey/enenlhet-dataset'
```

My Hugging Face account is configured to use SSH, so this error made no sense. It turned out that I had to do this, too, to configure SSH access for LFS in the *local* repository:

```bash
git config --local lfs.https://huggingface.co/.access ssh
```

That eliminated the authentication errors, but I still ran into the unfortunate problem that the `.git` directory was as large as the repository itself. That meant that I was trying to upload not just 10GB of data, but 10GB of Git information, too.

The solution was to use Hugging Face Hub commands in my notebook:

```python
# Log into Hugging Face Hub
from huggingface_hub import notebook_login
notebook_login()

whisper_dataset.push_to_hub(
    repo_id="enenlhet-asr/enenlhet-whisper-dataset",
    private=True,
    commit_message="Second commit of Enenlhet Whisper dataset"
)
```


