import typer

from zsl.image import process_classes

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


@cli.command("version")
def cli_version():
    print("Hello World")


if __name__ == "__main__":
    cli()
