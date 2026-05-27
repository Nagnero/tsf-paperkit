from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from tsf_paperkit.ablation.manager import run_ablations
from tsf_paperkit.checks.fairness import as_text, check_config_file, doctor_report
from tsf_paperkit.data.registry import load_dataset_registry, prepare_dataset
from tsf_paperkit.models.registry import list_models, prepare_model
from tsf_paperkit.reporting.table import write_table
from tsf_paperkit.runner.experiment import load_config, run_experiment

app = typer.Typer(help="Paper-ready TSF experiment harness.")
data_app = typer.Typer(help="Dataset recipe commands.")
model_app = typer.Typer(help="Model recipe commands.")
app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


@app.command()
def run(config: Path = typer.Option(..., "--config")):
    path = run_experiment(load_config(config))
    typer.echo(str(path))


@app.command()
def ablate(config: Path = typer.Option(..., "--config")):
    typer.echo(str(run_ablations(config)))


@app.command()
def table(input: Path = typer.Option(..., "--input"), format: str = typer.Option("markdown", "--format"), output: Path = typer.Option(..., "--output")):
    typer.echo(str(write_table(input, output, format)))


@app.command()
def check(config: Path = typer.Option(..., "--config"), format: str = typer.Option("text", "--format")):
    report = check_config_file(config)
    typer.echo(json.dumps(report, indent=2) if format == "json" else as_text(report))


@app.command()
def doctor(format: str = typer.Option("text", "--format")):
    report = doctor_report()
    typer.echo(json.dumps(report, indent=2) if format == "json" else as_text(report))


@data_app.command("list")
def data_list(registry: Path = typer.Option(Path("configs/dataset_registry.yaml"), "--registry")):
    typer.echo(yaml.safe_dump({"datasets": load_dataset_registry(registry)}, sort_keys=False))


@data_app.command("prepare")
def data_prepare(dataset: str = typer.Option(..., "--dataset"), cache_dir: Optional[str] = typer.Option(None, "--cache-dir"), registry: Path = typer.Option(Path("configs/dataset_registry.yaml"), "--registry")):
    typer.echo(json.dumps(prepare_dataset(dataset, cache_dir=cache_dir, registry_path=registry), indent=2))


@model_app.command("list")
def model_list(registry: Path = typer.Option(Path("configs/model_registry.yaml"), "--registry")):
    typer.echo(yaml.safe_dump({"models": list_models(registry)}, sort_keys=False))


@model_app.command("prepare")
def model_prepare(model: str = typer.Option(..., "--model"), cache_dir: Optional[str] = typer.Option(None, "--cache-dir"), registry: Path = typer.Option(Path("configs/model_registry.yaml"), "--registry")):
    typer.echo(json.dumps(prepare_model(model, cache_dir=cache_dir, registry_path=registry), indent=2))


if __name__ == "__main__":
    app()
