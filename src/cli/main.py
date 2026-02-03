from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.status import Status

app = typer.Typer(help="CLI para ingestão e busca de documentos.")
console = Console()


@dataclass
class AppConfig:
    ingest_source: str = "./data"
    search_top_k: int = 5
    status_verbose: bool = False


ENV_MAPPING = {
    "INGEST_SOURCE": "ingest_source",
    "SEARCH_TOP_K": "search_top_k",
    "STATUS_VERBOSE": "status_verbose",
}


def load_config(config_path: Path | None = None) -> AppConfig:
    config_data: Dict[str, Any] = {}

    if config_path is None:
        config_path = Path("config.yaml")

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config_data.update(yaml.safe_load(handle) or {})

    dotenv_path = Path(".env")
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        for env_key, config_key in ENV_MAPPING.items():
            env_value = os.getenv(env_key)
            if env_value is None:
                continue
            if config_key == "search_top_k":
                config_data[config_key] = int(env_value)
            elif config_key == "status_verbose":
                config_data[config_key] = env_value.lower() in {"1", "true", "yes"}
            else:
                config_data[config_key] = env_value

    return AppConfig(
        ingest_source=str(config_data.get("ingest_source", "./data")),
        search_top_k=int(config_data.get("search_top_k", 5)),
        status_verbose=bool(config_data.get("status_verbose", False)),
    )


@app.command()
def ingest(
    source: str = typer.Option(None, "--source", "-s", help="Pasta ou arquivo a ingerir."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Caminho do config.yaml."),
) -> None:
    """Ingere documentos usando configuração de ambiente."""
    settings = load_config(config)
    ingest_source = source or settings.ingest_source

    console.print(f"[bold blue]Iniciando ingestão[/bold blue] de {ingest_source}")
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("Processando documentos", total=100)
        for _ in range(100):
            progress.advance(task, 1)
    console.print("[green]Ingestão concluída com sucesso![/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Consulta de busca."),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Quantidade de resultados."),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Caminho do config.yaml."),
) -> None:
    """Executa uma busca e exibe os resultados."""
    settings = load_config(config)
    result_size = top_k or settings.search_top_k

    console.print(f"[bold cyan]Buscando[/bold cyan] '{query}' (top {result_size})")
    with Status("Consultando índice...", console=console):
        pass
    console.print("[green]Busca concluída.[/green]")
    for idx in range(1, result_size + 1):
        console.print(f"[yellow]Resultado {idx}[/yellow]: Documento exemplo {idx}")


@app.command()
def status(
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Caminho do config.yaml."),
) -> None:
    """Mostra o status atual do pipeline."""
    settings = load_config(config)

    console.print("[bold magenta]Status do pipeline[/bold magenta]")
    console.print(f"Origem de ingestão: [cyan]{settings.ingest_source}[/cyan]")
    console.print(f"Top-K padrão: [cyan]{settings.search_top_k}[/cyan]")
    if settings.status_verbose:
        console.print("[green]Modo verboso ativado.[/green]")


if __name__ == "__main__":
    app()
