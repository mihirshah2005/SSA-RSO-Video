"""Detection, tracking, classification and catalogue association of small
objects in Starship onboard video.

Four outputs are kept separate on purpose (see docs/DESIGN.md):

* detection       -- an image feature exists here, now
* tracking        -- consistent observations across frames
* classification  -- a supported physical category, or ``unknown``
* identification  -- a catalogue identity, only when the evidence is unique

A track number is never a satellite identity.
"""

__version__ = "0.1.0"
