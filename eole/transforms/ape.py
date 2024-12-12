import random
import yaml

from .transform import Transform, TransformConfig
from eole.transforms import register_transform
from eole.utils.logging import logger

from pydantic import Field


class APEConfig(TransformConfig):
    ape_token: str | None = Field(
        default="｟src_mt_sep｠", description="Separator between src and raw MT"
    )
    ape_inference_config: str | None = Field(
        default=None, description="Separator between src and raw MT"
    )
    ape_example_ratio: float | None = Field(
        default=0.5, description="Ratio of corpus to be used for APE. training"
    )


CACHE = {}


@register_transform(name="ape")
class APETransform(Transform):

    config_model = APEConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.ape_token = self.config.ape_token
        self.ape_example_ratio = self.config.ape_example_ratio
        self.ape_inference_config = self.config.ape_inference_config

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
            ape_inference_config.model_path = [self.full_config.training.train_from]
            ape_inference_config.seed = -1
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
                for i, (_ex, _transform_pipe, _cid) in enumerate(batch):
                    # logger.info(_cid)
                    # logger.info(_ex)
                    prefix_dict = self._get_prefix_dict(_transform_pipe, _cid)
                    src.append(prefix_dict["src"] + " ".join(_ex["src"]))

                _, _, preds = self.mt_engine.infer_list(src)
                self.mt_engine.terminate()
                preds = [item for sublist in preds for item in sublist]  # flatten

                # logger.info("srcs")
                # logger.info(src)
                # logger.info("preds")
                # logger.info(preds)
                # logger.info("refs")
                # logger.info([' '.join(_ex['tgt'])for i, (_ex, _, _) in enumerate(batch)
                #              if i in ape_indices])

                for i, _pred in enumerate(preds):
                    ex, transform_pipe, corpus_name = batch[i]
                    ex["src"] = (" ".join(ex["src"]) + self.ape_token + _pred).split()
                    batch[i] = (ex, transform_pipe, corpus_name)
        return batch
