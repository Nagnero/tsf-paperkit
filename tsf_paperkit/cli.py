from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml

from tsf_paperkit.ablation.manager import run_ablations
from tsf_paperkit.checks.fairness import as_text, check_config_file, doctor_report
from tsf_paperkit.data.registry import dataset_recipe, list_dataset_recipes, prepare_dataset
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
def data_list(
    registry: Path = typer.Option(Path("configs/dataset_registry.yaml"), "--registry"),
    family: Optional[str] = typer.Option(None, "--family"),
    status: Optional[str] = typer.Option(None, "--status"),
    json_output: bool = typer.Option(False, "--json"),
):
    recipes = list_dataset_recipes(registry, family=family, status=status)
    payload = {"datasets": recipes}
    typer.echo(json.dumps(payload, indent=2) if json_output else yaml.safe_dump(payload, sort_keys=False))


@data_app.command("inspect")
def data_inspect(
    dataset: str = typer.Option(..., "--dataset"),
    registry: Path = typer.Option(Path("configs/dataset_registry.yaml"), "--registry"),
    json_output: bool = typer.Option(False, "--json"),
):
    recipe = dataset_recipe(dataset, registry)
    payload = {"dataset": recipe}
    typer.echo(json.dumps(payload, indent=2) if json_output else yaml.safe_dump(payload, sort_keys=False))


@data_app.command("prepare")
def data_prepare(
    dataset: Optional[str] = typer.Option(None, "--dataset"),
    family: Optional[str] = typer.Option(None, "--family"),
    cache_dir: Optional[str] = typer.Option(None, "--cache-dir"),
    registry: Path = typer.Option(Path("configs/dataset_registry.yaml"), "--registry"),
    force: bool = typer.Option(False, "--force"),
    confirm_family: bool = typer.Option(False, "--confirm-family"),
    json_output: bool = typer.Option(False, "--json"),
):
    if not dataset and not family:
        raise typer.BadParameter("select exactly one dataset with --dataset, or explicitly opt into a family with --family")
    if dataset and family:
        raise typer.BadParameter("use either --dataset or --family, not both")
    if dataset:
        result = prepare_dataset(dataset, cache_dir=cache_dir, registry_path=registry, force=force)
        typer.echo(json.dumps(result, indent=2) if json_output else yaml.safe_dump(result, sort_keys=False))
        return

    recipes = list_dataset_recipes(registry, family=family)
    if not recipes:
        raise typer.BadParameter(f"unknown dataset family: {family}")
    selected = [r for r in recipes if r.get("status") == "supported"]
    skipped = [r for r in recipes if r.get("status") != "supported"]
    results = []
    if confirm_family:
        results = [prepare_dataset(r["name"], cache_dir=cache_dir, registry_path=registry, force=force) for r in selected]
    payload = {
        "status": "ready" if results and all(r["status"] == "ready" for r in results) else "preview",
        "family": family,
        "requires_confirmation": bool(selected) and not confirm_family,
        "preview": [r["name"] for r in recipes],
        "prepared": results,
        "skipped": [{"dataset": r["name"], "status": r.get("status"), "reason": r.get("skip_reason") or r.get("blocked_reason")} for r in skipped],
    }
    typer.echo(json.dumps(payload, indent=2) if json_output else yaml.safe_dump(payload, sort_keys=False))


@model_app.command("list")
def model_list(registry: Path = typer.Option(Path("configs/model_registry.yaml"), "--registry")):
    typer.echo(yaml.safe_dump({"models": list_models(registry)}, sort_keys=False))


@model_app.command("prepare")
def model_prepare(model: str = typer.Option(..., "--model"), cache_dir: Optional[str] = typer.Option(None, "--cache-dir"), registry: Path = typer.Option(Path("configs/model_registry.yaml"), "--registry")):
    typer.echo(json.dumps(prepare_model(model, cache_dir=cache_dir, registry_path=registry), indent=2))


if __name__ == "__main__":
    app()
