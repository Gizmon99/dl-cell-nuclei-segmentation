"""
This is a boilerplate test file for pipeline 'model_train'
generated using Kedro 0.18.6.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from pathlib import Path

import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager


@pytest.fixture
def config_loader():
    return ConfigLoader(conf_source=str(Path.cwd() / settings.CONF_SOURCE))


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="cell_nuclei_segmentation",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )

class TestModelTrain:

    def test_model_performace(self, project_context):
        dataset = project_context.catalog.load('dataset')
        model = project_context.catalog.load('lightning_model')
        accuracy = model.eval(dataset)
        assert accuracy >= 0.8