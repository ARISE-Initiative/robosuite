# Using the indicator object
When you are exploring an environment, it can be very helpful to have a free object moving wherever you want. This can be done by providing the `use_indicator_object` flag on environment initialization.

```python
env = SawyerLiftEnv(
    ...,
    use_indicator_object=True
)
```

The indicator object is a red ball. Its location is specified as (x, y, z) coordinates in the word frame. It is initially located at (0,0,0) and can be moved by calling

```python
env.move_indicator([x, y, z])
```

Example:
![indicator object](../resources/indicator_object.png)
