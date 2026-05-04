"""Single CLI entrypoint for the EPP Reconciliation AI Agent."""
import json
import logging
import sys
from pathlib import Path

import typer
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def reconcile(
    input_path: str = typer.Option(..., "--input", "-i", help="Path to IC_InputData.xlsx"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    amount_tolerance: float = typer.Option(None, help="Override .env AMOUNT_TOLERANCE"),
    date_tolerance_days: int = typer.Option(None, help="Override .env DATE_TOLERANCE_DAYS"),
    fuzzy_threshold: int = typer.Option(None, help="Override .env FUZZY_NARRATION_THRESHOLD"),
    skip_ai: bool = typer.Option(False, "--skip-ai", help="Skip all AI calls"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    try:
        _run(input_path, output_dir, amount_tolerance, date_tolerance_days,
             fuzzy_threshold, skip_ai, verbose)
    except Exception as exc:
        console.print(f"\n[bold red]ERROR:[/bold red] {exc}")
        if verbose:
            import traceback
            console.print_exception()
        raise SystemExit(1)


def _run(input_path, output_dir, amount_tolerance, date_tolerance_days,
         fuzzy_threshold, skip_ai, verbose):

    # ── 1. Config ────────────────────────────────────────────────────────────
    from src.config import Config
    overrides = {}
    if amount_tolerance is not None:
        overrides["amount_tolerance"] = amount_tolerance
    if date_tolerance_days is not None:
        overrides["date_tolerance_days"] = date_tolerance_days
    if fuzzy_threshold is not None:
        overrides["fuzzy_narration_threshold"] = fuzzy_threshold
    if skip_ai:
        overrides["skip_ai"] = True

    config = Config(**overrides)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
        stream=sys.stderr,
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console, transient=True) as progress:

        # ── 2. Load ──────────────────────────────────────────────────────────
        task = progress.add_task("Loading data…", total=None)
        from src.data.loader import load_data
        df1, df2 = load_data(input_path)
        progress.update(task, description=f"Loaded {len(df1)} + {len(df2)} rows")

        # ── 3. Clean ─────────────────────────────────────────────────────────
        progress.update(task, description="Cleaning data…")
        from src.data.cleaner import normalize_amounts, normalize_dates, normalize_narration
        for fn in [normalize_amounts, normalize_dates, normalize_narration]:
            df1 = fn(df1)
            df2 = fn(df2)

        # ── 4. Validate ──────────────────────────────────────────────────────
        progress.update(task, description="Validating data…")
        from src.data.validator import validate
        df_combined_for_val = pd.concat([df1, df2], ignore_index=True)
        val_report = validate(df_combined_for_val)

        if not val_report.passed:
            if val_report.duplicates:
                console.print(f"[yellow]⚠ {len(val_report.duplicates)} duplicate row(s) found[/yellow]")
            if val_report.currency_issues:
                for ci in val_report.currency_issues:
                    console.print(f"[yellow]⚠ Currency mismatch: {ci}[/yellow]")

        # ── 5. Match ─────────────────────────────────────────────────────────
        progress.update(task, description="Running matching engine…")
        from src.engine.orchestrator import run_reconciliation
        report = run_reconciliation(df1, df2, config)

        # Build df_all for exporter
        df_all = pd.concat([df1.assign(side="A"), df2.assign(side="B")], ignore_index=True)

        # ── 6. AI ─────────────────────────────────────────────────────────────
        if not config.skip_ai:
            from src.agent.agent import analyze_unmatched, score_complex_match, generate_summary

            progress.update(task, description="AI: analysing unmatched transactions…")
            unmatched_tx = []
            for idx in report.unmatched_transactions:
                if idx in df_all.index:
                    row = df_all.loc[idx].to_dict()
                    row["index"] = idx
                    unmatched_tx.append(row)
            report.anomaly_analysis = analyze_unmatched(unmatched_tx, config)

            progress.update(task, description="AI: scoring complex groups…")
            judgments = []
            for grp in report.complex_groups:
                j = score_complex_match(grp, df_all, config)
                judgments.append(j)
            report.complex_judgments = judgments

            progress.update(task, description="AI: generating executive summary…")
            report.executive_summary = generate_summary(report, config)
        else:
            from src.agent.agent import _fallback_summary
            report.executive_summary = _fallback_summary(report)

        # ── 7. Export ────────────────────────────────────────────────────────
        progress.update(task, description="Exporting results…")
        from src.output.exporter import export_results
        xlsx_path, json_path = export_results(df_all, report, output_dir, config)

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="EPP Reconciliation Summary", show_header=True, header_style="bold blue")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Transactions",     str(report.total_transactions))
    table.add_row("Matched Pairs",          str(len(report.matched_pairs)))
    table.add_row("Complex Groups (R8)",    str(len(report.complex_groups)))
    table.add_row("Unmatched Transactions", str(len(report.unmatched_transactions)))
    table.add_row("Match Rate",             f"{report.match_rate_pct:.1f}%")
    table.add_row("Excel Report",           xlsx_path)
    table.add_row("Audit JSON",             json_path)
    console.print(table)


if __name__ == "__main__":
    app()
