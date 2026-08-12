"""Who the person is, apart from their face.

A face photo carries a face and nothing else. Everything the viewer actually
reads as "this is me" — build, gender presentation, hair, clothing, how they
hold themselves — is invented by the image model unless it is specified. That
is the whole reason an unconstrained run returns a plausible stranger in a grey
t-shirt: not a failure, an unfilled spec.

So the subject is an explicit input, alongside the face, and it travels through
the pipeline in two forms, because they are obeyed differently. Measured live on
kontext against the same reference face:

    clothing/accessories as TEXT   -> obeyed (red hoodie, beanie, yellow shoes
                                      all appeared as written)
    build as TEXT                  -> IGNORED ("heavy-set and broad" changed
                                      nothing; an editing model keeps the body
                                      it was given)
    and the text cost identity     -> face drift 0.109 -> 0.295, nearly tripled

    build + clothing as a REFERENCE IMAGE on a multi-reference model
    (nanobanana, face ref + body ref) -> both obeyed, face drift 0.176

Hence the rule this module encodes: **text for what a model will restyle,
a reference image for what it will not.** Build is the clearest case — if you
need a body that is not the one in the face photo, you need `body_ref`, because
no adjective will do it. `kontext` cannot take one at all (max_reference_images
is 1), which is why supplying `body_ref` switches the start-frame model.

Nothing here is guessed about the person: `from_photo` fills only what can be
measured off the photo, and leaves the rest empty rather than inventing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

#: Start-frame model when only a face is supplied — an editor, 1 reference.
SINGLE_REF_MODEL = "kontext"
#: Start-frame model when a body/outfit reference is supplied too. Chosen by
#: measurement, not preference: at 3 references it kept the face (drift 0.176)
#: while taking build and clothing from the second image. `seedream5` accepts
#: 14 references but lost the face outright on the same inputs (drift 0.752).
MULTI_REF_MODEL = "nanobanana"


@dataclass(frozen=True)
class Subject:
    """The person's spec: everything the face photo does not carry.

    Every field is optional and empty means "not specified" — an empty field is
    left out of the prompt entirely rather than filled with a default, because
    a default here is a silent decision about what someone looks like.
    """

    #: How the person presents. Written into the prompt verbatim, so use the
    #: words you want the model to read ("man", "woman", "androgynous").
    gender: str = ""
    #: e.g. "late 30s". A band, not a number — models read bands better.
    age: str = ""
    #: e.g. "lean and athletic", "heavy-set and broad", "slight, narrow-framed".
    #: TEXT ALONE DOES NOT MOVE THIS on an editing model — see `body_ref`.
    build: str = ""
    hair: str = ""
    skin: str = ""
    #: e.g. "black compression tee, grey running shorts". Text works well here.
    outfit: str = ""
    footwear: str = ""
    #: How they hold themselves in the still, e.g. "knees soft, arms out".
    posture: str = ""
    #: Anything else that belongs in the description, verbatim.
    extra: str = ""

    #: Path to a photo showing the BODY (and usually the clothing) to copy. The
    #: only reliable way to set build. Its presence switches the start frame to
    #: MULTI_REF_MODEL, because the single-reference editor cannot accept it.
    body_ref: str = ""

    #: Path to a photo showing the POSE to reproduce — limb positions, weight,
    #: camera angle. Same reasoning as `body_ref`: a pose is geometry, and
    #: geometry does not survive being described in adjectives.
    #:
    #: It contributes POSE ONLY. The face always comes from the face photo, and
    #: the prompt says so explicitly, so a pose reference that happens to show
    #: someone else does not put that person's likeness in the output. Use a
    #: clean reference: watermarks and captions in a reference image get
    #: reproduced into the generated frame along with everything else.
    pose_ref: str = ""

    def to_prompt(self) -> str:
        """The subject as one description, in a stable order.

        Order is fixed so that two runs of the same subject differ only where
        the subject differs — a prompt that reshuffles is a prompt whose output
        cannot be compared across attempts.
        """
        person = " ".join(p for p in (self.age, self.build, self.gender) if p)
        parts = [f"The person is {person}." if person else ""]
        if self.hair:
            parts.append(f"Hair: {self.hair}.")
        if self.skin:
            parts.append(f"Skin: {self.skin}.")
        worn = ", ".join(p for p in (self.outfit, self.footwear) if p)
        if worn:
            parts.append(f"Wearing {worn}.")
        if self.posture:
            parts.append(f"Posture: {self.posture}.")
        if self.extra:
            parts.append(self.extra.strip())
        return " ".join(p for p in parts if p).strip()

    @property
    def specified(self) -> tuple[str, ...]:
        """Which fields the caller actually set — what a gate may check."""
        return tuple(f for f in ("gender", "age", "build", "hair", "skin",
                                 "outfit", "footwear", "posture")
                     if getattr(self, f))

    @property
    def reference_roles(self) -> tuple[tuple[str, str], ...]:
        """(path, role) for every reference image, face first.

        Position is the contract: the prompt names references by ordinal, and
        the URLs go over the wire `|`-joined in this order, so face must stay
        first and the sequence must match what `reference_clause` describes.
        """
        roles = [("__face__", "the person's face and identity")]
        if self.body_ref:
            roles.append((self.body_ref, "the person's body build and clothing"))
        if self.pose_ref:
            roles.append((self.pose_ref, "the pose and camera angle only"))
        return tuple(roles)

    def reference_clause(self) -> str:
        """Tell the model what each reference image is FOR, by position.

        Naming roles positionally is what made a multi-reference generation
        keep the face from one image and take build/clothing from another
        instead of blending them.
        """
        refs = self.reference_roles
        if len(refs) < 2:
            return ""
        ordinals = ("FIRST", "SECOND", "THIRD", "FOURTH")
        parts = [f"the {ordinals[i]} image for {role}"
                 for i, (_, role) in enumerate(refs) if i < len(ordinals)]
        return ("Use " + ", and ".join(parts)
                + ". The face must come only from the first image.")

    def start_model(self, default: str = SINGLE_REF_MODEL) -> str:
        """Which image model can honour this subject."""
        return MULTI_REF_MODEL if len(self.reference_roles) > 1 else default

    @classmethod
    def from_photo(cls, face_photo: str | Path, **overrides) -> "Subject":
        """Fill only what is measurable off the photo; leave the rest empty.

        Reads apparent sex and age from the face with the local InsightFace
        estimator, so the prompt at least stops contradicting the reference. It
        is a coarse estimator, and these are ITS labels for the photo, not
        ground truth about a person — anything the caller passes in `overrides`
        wins, and callers who know the answer should say so.
        """
        from .identity_arcface import face_attributes

        attrs = face_attributes(face_photo)
        fields: dict[str, str] = {}
        if attrs:
            if attrs.get("sex") in ("M", "F"):
                fields["gender"] = "man" if attrs["sex"] == "M" else "woman"
            age = attrs.get("age")
            if isinstance(age, (int, float)):
                fields["age"] = _age_band(int(age))
        fields.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**fields)


def _age_band(age: int) -> str:
    """An age band, because a decade reads better to a model than '43'."""
    if age < 20:
        return "in their teens"
    decade, rem = (age // 10) * 10, age % 10
    third = "early" if rem <= 3 else ("late" if rem >= 7 else "mid")
    return f"in their {third} {decade}s"


#: No subject specified: the model invents everything but the face. Kept as a
#: name so call sites read as a deliberate choice rather than an omission.
UNSPECIFIED = Subject()
