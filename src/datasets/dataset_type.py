import enum


class DatasetType(enum.Enum):
    """Supported datasets, with a plot colour and canonical ordering."""

    MESA = "MESA"
    CFS = "CFS"
    HOMEPAP = "HomePAP"

    @property
    def colour(self):
        return {
            DatasetType.MESA: "#a3c4f3",
            DatasetType.CFS: "#b5e6b5",
            DatasetType.HOMEPAP: "#f0d6a4",
        }[self]

    @classmethod
    def ordered(cls, available):
        """Return datasets in canonical order, with extras sorted at the end."""
        order = [
            DatasetType.MESA.value,
            DatasetType.CFS.value,
            DatasetType.HOMEPAP.value,
        ]
        return [d for d in order if d in available] + sorted(available - set(order))
