from evaluate import evaluator
from datasets import load_dataset
from transformers import AutoFeatureExtractor, pipeline

# package evaluate build from source, in case there are internet connection errors
eval_path = '/home/arda/yi/evaluate/metrics/accuracy'
vit_model_path = '/mnt/disk1/models/vit-base-patch16-224'
data_files = {'validation': ['/mnt/disk1/datasets/imagenet_val/**']}
cache_dir = '/mnt/disk1/datasets/'


def test_original_model():
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(vit_model_path)

    preprocessor = AutoFeatureExtractor.from_pretrained(vit_model_path)
    vanilla_vit = pipeline("image-classification", model=model, feature_extractor=preprocessor, batch_size=512)

    e = evaluator("image-classification")
    eval_dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=cache_dir, task="image-classification", ignore_verifications=True)
    print('start evaluation...')
    results = e.compute(
        model_or_pipeline=vanilla_vit,
        data=eval_dataset['validation'],
        metric=eval_path,
        input_column="image",
        label_column="labels",
        label_mapping=model.config.label2id,
        strategy="simple",
    )
    print(f"Vanilla model: {results['accuracy']*100:.2f}%") # 80.25%


def test_transformer_api_cpu_acc():
    from bigdl.llm.transformers import AutoModelForImageClassification

    model = AutoModelForImageClassification.from_pretrained(
        vit_model_path,
        load_in_4bit=True,
    )

    preprocessor = AutoFeatureExtractor.from_pretrained(vit_model_path)
    vanilla_vit = pipeline("image-classification", model=model, feature_extractor=preprocessor, batch_size=512)

    e = evaluator("image-classification")
    eval_dataset = load_dataset("imagefolder", data_files=data_files, cache_dir=cache_dir, task="image-classification", ignore_verifications=True)
    print('start evaluation...')
    results = e.compute(
        model_or_pipeline=vanilla_vit,
        data=eval_dataset['validation'],
        metric=eval_path,
        input_column="image",
        label_column="labels",
        label_mapping=model.config.label2id,
        strategy="simple",
    )
    print(f"4bit quantize model: {results['accuracy']*100:.2f}%") # 80.08%, 4.09%


if __name__ == "__main__":
    # test original model
    # test_original_model()

    # test model using transformer api
    test_transformer_api_cpu_acc()
