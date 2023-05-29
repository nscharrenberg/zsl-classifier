import typer

from zsl.image import process_classes
from zsl.text import main as text_main
from zsl.clustering import embedded_clustering, llm_clustering

cli = typer.Typer()


@cli.command("image")
def cli_image(
        data: str = typer.Option(None,
                                 help="The directory containing the classes.txt and directory named \"images\" containing the image data"),
        whitelist: str = typer.Option(None,
                                      help="Only take the classes that are in this list"),
        verbose: bool = typer.Option(False,
                                     help="If True, then intermediate console message will be displayed to indicate "
                                          "progress."),
):
    process_classes(path=data, whitelist=whitelist, verbose=verbose)


@cli.command("cluster")
def cli_image(
        seen: str = typer.Option("../datasets/seen.csv",
                                 help="Path to the seen dataset"),
        unseen: str = typer.Option("../datasets/unseen.csv",
                                   help="Path to the unseen dataset"),
        header: str = typer.Option("disease",
                                   help="Path to the unseen dataset"),
        confusion: bool = typer.Option(False,
                                       help="Display a confusion matrix"),
        approach: str = typer.Option("embedding",
                                       help="The type of clustering to use (embedding, llm)."),
        verbose: bool = typer.Option(False,
                                     help="If True, then intermediate console message will be displayed to indicate "
                                          "progress."),
):
    if approach == "embedding":
        embedded_clustering(seen, unseen, header, confusion, verbose)
    elif approach == "llm":
        print(llm_clustering(seen, unseen, header, confusion, verbose))
    else:
        raise Exception(f"Invalid Approach \"{approach}\".")


@cli.command("text")
def cli_text():
    text_main()


@cli.command("version")
def cli_version():
    print("0.1.0")


if __name__ == "__main__":
    cli()
