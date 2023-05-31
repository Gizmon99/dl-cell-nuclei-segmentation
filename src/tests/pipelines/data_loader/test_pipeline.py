"""
This is a boilerplate test file for pipeline 'data_loader'
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
from kedro.io import DataCatalog
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
import os


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

class TestDataLoading:

    def test_dvc(self, project_context):
        os.system('dvc pull data/01_raw/test_dataset')
        raw_images = os.listdir(project_context.project_path / 'data/01_raw/test_dataset/rawimages')
        target_masks = os.listdir(project_context.project_path / 'data/01_raw/test_dataset/groundtruth')
        assert 'Ganglioneuroblastoma_0.tif' in raw_images
        assert 'Neuroblastoma_0.tif' in raw_images
        assert 'normal_0.tif' in raw_images
        assert 'otherspecimen_0.tif' in raw_images
        assert raw_images == target_masks
