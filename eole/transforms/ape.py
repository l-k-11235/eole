import difflib
import editdistance
import Levenshtein
import random
import spacy
import yaml

from .transform import Transform, TransformConfig
from eole.transforms import register_transform
from eole.utils.logging import logger

from pydantic import Field


CACHE = {}

SPACY_MODELS = {"fr": spacy.load("fr_core_news_sm"), "en": spacy.load("en_core_web_sm")}


def compute_correction_rate(ex, tgt_lang):
    hyp_name = "ref"
    for _k in ("raw_mt", hyp_name):
        ex[_k] = ex[_k].replace("'", "’")

    doc = SPACY_MODELS[tgt_lang](ex[hyp_name])
    hyp_tokens = [tok.text for tok in doc]
    doc = SPACY_MODELS[tgt_lang](ex["raw_mt"])
    raw_mt_tokens = [tok.text for tok in doc]

    differ = difflib.Differ()

    # Compute differences
    diff_result = list(differ.compare(raw_mt_tokens, hyp_tokens))
    corrections = list([_diff for _diff in diff_result if _diff[0] in ["+", "-"]])
    # print(diff_result)
    correction_rate = len(corrections) / len(diff_result)
    ex[hyp_name + "_corrections"] = corrections
    ex[hyp_name + "_correction_rate"] = correction_rate
    ex[hyp_name + "LevenshteinDistance"] = Levenshtein.distance(
        ex["raw_mt"], ex[hyp_name]
    )
    return ex


def calculate_ter(reference: str, hypothesis: str, lang: str) -> float:
    """
    Calculate the Translation Error Rate (TER) for a single sentence.

    Args:
        reference (str): The reference sentence.
        hypothesis (str): The machine-translated sentence.

    Returns:
        float: The TER score.
    """
    # Tokenize sentences into words
    doc = SPACY_MODELS[lang](reference)
    ref_tokens = [tok.text for tok in doc]
    doc = SPACY_MODELS[lang](hypothesis)
    hyp_tokens = [tok.text for tok in doc]
    # Calculate the number of edits
    num_edits = editdistance.eval(ref_tokens, hyp_tokens)
    # Calculate TER
    ter = num_edits / len(ref_tokens) if len(ref_tokens) > 0 else float("inf")
    return ter


def TER_stats_vector(record, lang_tgt, with_strings=False):
    if record["ref"] is not None:
        ter = calculate_ter(
            reference=record["ref"], hypothesis=record["raw_mt"], lang=lang_tgt
        )

        record = compute_correction_rate(record, lang_tgt)
        vector = {
            "hyp_length": len(record["raw_mt"]),
            "hyp_ter": 100 * ter,
            "ref_correction_rate": 100 * record["ref_correction_rate"],
            "ref_corrections": record["ref_corrections"],
        }
        if with_strings:
            for _name in ["source", "raw_mt", "ref"]:
                vector[_name] = record[_name]
    else:
        vector = None
    return vector


class APEConfig(TransformConfig):
    ape_token: str | None = Field(
        default="｟src_mt_sep｠", description="Separator between src and raw MT"
    )
    ape_inference_config: str | None = Field(default=None, description="")
    ape_mt_checkpoint: str | None = Field(default=None, description="")
    ape_example_ratio: float | None = Field(
        default=0.5, description="Ratio of corpus to be used for APE. training"
    )
    ape_bounds_dict: dict | None = Field(default=None, description="")
    ape_spacy_lang_map: dict | None = Field(default=None, description="")
    ape_logging: bool | None = Field(default=False, description="")
    ape_batch_type: str | None = Field(
        default="sents", description="Batch type for mt-engine."
    )
    ape_batch_size: int | None = Field(
        default=10, description="Batch size for mt-engine."
    )


@register_transform(name="ape")
class APETransform(Transform):

    config_model = APEConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.ape_token = self.config.ape_token
        self.ape_example_ratio = self.config.ape_example_ratio
        self.ape_inference_config = self.config.ape_inference_config
        self.ape_mt_checkpoint = self.config.ape_mt_checkpoint
        self.ape_bounds_dict = self.config.ape_bounds_dict
        self.ape_spacy_lang_map = self.config.ape_spacy_lang_map
        self.ape_logging = self.config.ape_logging
        self.ape_batch_type = self.config.ape_batch_type
        self.ape_batch_size = self.config.ape_batch_size

    def _get_prefix_dict(self, transform_pipe, corpus_id):
        prefix_transform = [
            _transform
            for _transform in transform_pipe.transforms
            if _transform.name == "prefix"
        ][0]
        prefix_dict = {"src": "", "tgt": ""}
        if prefix_transform:
            prefix_dict = prefix_transform.prefix_dict.get(corpus_id)
        # logger.info("# prefix_dict")
        # logger.info(prefix_dict)
        return prefix_dict

    def maybe_load_engine(self):
        if CACHE.get("engine", None) is None:
            logger.info("# loading MT engine")
            from eole.config.run import PredictConfig
            from eole.inference_engine import InferenceEnginePY

            with open(self.ape_inference_config) as f:
                ape_inference_config = yaml.safe_load(f.read())
            ape_inference_config = PredictConfig(**ape_inference_config)
            # set_random_seed(ape_inference_config.seed, use_gpu(ape_inference_config))
            ape_inference_config.model_path = [self.ape_mt_checkpoint]
            ape_inference_config.seed = -1
            ape_inference_config.batch_type = self.ape_batch_type
            ape_inference_config.batch_size = self.ape_batch_size
            CACHE["engine"] = InferenceEnginePY(ape_inference_config)
        self.mt_engine = CACHE["engine"]

    # @classmethod
    # def get_specials(cls, config):
    #     """Add the APE tokens to the src vocab."""
    #     src_specials = list()
    #     logger.info('config')
    #     logger.info(config)
    #     src_specials.extend(
    #         [config.ape_token]
    #     )
    #     return (src_specials, list())

    def _filter(self, example_stats):
        to_filter = False
        if example_stats is not None:
            for _key in self.ape_bounds_dict.keys():
                if not to_filter:
                    if (
                        example_stats[_key] < self.ape_bounds_dict[_key][0]
                        or example_stats[_key] > self.ape_bounds_dict[_key][1]
                    ):
                        to_filter = True
        return to_filter

    def filter_bucket(self, synthetic_examples, lang_tgt, with_stats=False):
        # Compute stats
        stats = [
            TER_stats_vector(_ex, lang_tgt, with_strings=False)
            for _ex in synthetic_examples
        ]
        keep = []
        delete = []
        for i, _ex in enumerate(synthetic_examples):
            if with_stats and stats[i] is not None:
                _ex.update(stats[i])
            if self._filter(stats[i]):
                delete.append(i)
            else:
                keep.append(i)
        return keep, delete

    def batch_apply(self, batch, is_train=False, stats=None, **kwargs):
        # One sub bucket per corpora
        (_, _, corpora_name) = batch[0]
        # avoid using a corpus with a synthetic target
        # logger.info(f'{corpora_name} {".KD" not in corpora_name}')

        if ".KD" not in corpora_name:
            random_float = random.uniform(0, 1)
            if random_float < self.ape_example_ratio:

                # logger.info(f'Learn APE task on this sub bucket of size {len(batch)} ({random_float} < {self.ape_example_ratio})') # noqa
                self.maybe_load_engine()

                src = []
                new_batch = []
                _, transform_pipe, cid = batch[0]  # 1 dataset per sub bucket
                prefix_dict = self._get_prefix_dict(transform_pipe, cid)
                if self.ape_logging:
                    logger.info(f"Dataset: {cid} ({len(batch)}) examples")
                for i, (_ex, _, _) in enumerate(batch):
                    src.append(prefix_dict["src"] + " ".join(_ex["src"]))
                _, _, preds, bucket = self.mt_engine.infer_list(src)
                self.mt_engine.terminate()
                preds = [item for sublist in preds for item in sublist]  # flatten
                if self.ape_bounds_dict is not None and corpora_name != "valid":
                    synthetic_examples = []
                    for i, (_ex, _, _) in enumerate(batch):
                        synthetic_examples.append(
                            {
                                "ref": " ".join(_ex["tgt"])
                                if _ex["tgt"] is not None
                                else None,
                                "raw_mt": preds[i],
                            }
                        )
                    if self.ape_logging:
                        logger.info(f"# synthetic_examples: {synthetic_examples}")
                    lang_tgt = self.ape_spacy_lang_map[prefix_dict["src"]]
                    keep, _ = self.filter_bucket(
                        synthetic_examples, lang_tgt, with_stats=True
                    )
                    if self.ape_logging:
                        logger.info(
                            f"keep: {len(keep)} examples ({100 * len(keep) / len(batch)} %)"
                        )  # noqa
                        logger.info(keep)
                else:
                    logger.info("No boundaries found for TER statistics.")
                    logger.info(
                        "No filter was applied to the synthetic examples for the artificial APE."
                    )  # noqa
                    keep = [i for i, _ in enumerate(batch)]
                for i in keep:
                    ex, transform_pipe, corpus_name = batch[i]
                    ex["src"] = (
                        preds[i] + self.ape_token + " ".join(ex["src"])
                    ).split()
                    new_batch.append((ex, transform_pipe, corpus_name))
                batch = new_batch
        return batch
