# Fields

<!-- help: Fields -->

## Copy fields to another runs folder

**Fields and settings → Fields → Export fields** saves the field
definitions, with their options and the order of the tag options, to
`tensorboard-book-fields.json` (format `tensorboard-book-fields`). Values
are not included. **Import fields** reads that file, an annotations file
(only its fields are used) or a version 1 file with properties. Fields are
matched by name: those that already exist are kept as they are, or updated
from the file if you choose **Update from the file**. Updating can change a
field's type. Values are never touched.

## Coming from properties

Before fields, runs had *properties*, which were either a number or a
category. The first time a database with properties is opened, it is
upgraded:

- a number property becomes a number field.
- a category property becomes a choice field whose options are its values
  in alphabetical order, so pill colors start out the same.
- before anything changes, the annotations file is copied to
  `annotations-v1.json` next to it, so the previous state is kept.

Annotation files with properties (version 1) can still be imported.
