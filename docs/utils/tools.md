# tools/ reference

## convert_legacy_checkpoint.py

Reads a checkpoint written by an older release and writes one the current trainer can
resume. It does not change the weights, only the container.

```bash
python tools/convert_legacy_checkpoint.py old.pt new.pt
```

## demo_clips.py

Cuts the demonstration clips from source footage and writes the GIFs used on the landing
page. It needs the source videos, which are not in the repository.

## screenshots.py

Captures the three windows in their default state and writes the PNGs referenced by the
GUI pages. Run it with the application on screen; it does not fake a window.

## Adding a tool

A script belongs in `tools/` when it is for maintainers: it is not part of the installed
package, and it may depend on things the runtime does not.
