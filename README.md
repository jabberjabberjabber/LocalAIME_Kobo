##Info

**This version has been modified to work natively with KoboldCpp.**

Temperature, min_p, top_p, rep_pen, and top_k can be set by editing /src/utils/llm.py.


# LocalAIME

This simple tool tests local (or not) LLMs on the AIME problems. Even if some models are specifically trained to solve AIME-style problems or even trained specifically on some of them (by accident or purpose), it is still useful for comparing models of the same family or different quantizations of the same exact model. It would also be interesting to test same model, same quantization, but from different sources on huggingfcace.

## Example results

<div align="center">
    <img alt="Plot image" src="media/accuracy_comparison.png" width="70%">
    </br>
    <img alt="Plot image" src="media/performance_heatmap.png" width="90%">
</div>

## Setup

First of all prepare the project for the first test:
```sh
git clone https://github.com/Belluxx/LocalAIME.git
cd LocalAIME
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Run benchmark

Now you are ready to test a model on AIME 2024. Be sure to match both the `--base-url` and `--model` identifier based on which platform and which exact model you are using.

### KoboldCpp
```sh
python3 src/main.py \
    --base-url 'http://127.0.0.1:5001/v1' \
    --model 'gemma3:4b' \
    --max-tokens 32000 \
    --timeout 2000 \
    --problem-tries 3
```

### See results

After the test is finished, you can open the generated `model-name.json` file and check the results.

If you test many models you can also put all of them in a directory (eg. `results/`) and plot the results to get an overview:
```sh
python3 src/plot.py results
```

Then check the plots inside `plots/`

## Credits

AIME 2024 problems dataset retrieved from [HuggingFaceH4](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
