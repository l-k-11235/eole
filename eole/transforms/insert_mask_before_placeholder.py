from eole.constants import DefaultTokens
from eole.transforms import register_transform
from eole.utils.logging import logger
from .transform import Transform, TransformConfig
from pydantic import Field
from typing import List


class InsertMaskBeforePlaceholderConfig(TransformConfig):
    response_patterns: List[str] | None = Field(
        default=["Response : ｟newline｠"],
        description="Response pattern to locate the end of the prompt.",
    )


@register_transform(name="insert_mask_before_placeholder")
class InsertMaskBeforePlaceholdersTransform(Transform):
    """Add the `DefaultTokens.MASK_BEFORE` placeholder between
    the prompt and the response in an LM finetuning exemple.
    This is necessary to enable the 'zero-out prompt loss' mechanism.
    """

    config_model = InsertMaskBeforePlaceholderConfig

    def __init__(self, config):
        super().__init__(config)

    def _parse_config(self):
        self.response_patterns = self.config.response_patterns

    def apply(self, example, is_train=False, stats=None, **kwargs):
        _src = " ".join(example["src"])
        response = None
        for _pattern in self.response_patterns:
            if len(_src.split(_pattern)) == 2:
                prompt, response = _src.split(_pattern)
                response = DefaultTokens.MASK_BEFORE.join([_pattern, response])
        if response is not None:
            _src = "".join([prompt, response])
            example["src"] = _src.split(" ")
            example["tgt"] = _src.split(" ")
        else:
            logger.info("The mask_before could not be inserted")
            return example
        return example
