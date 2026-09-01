"""Javni API za pokretanje VizTool prikaza."""
from model_visualizer.analyzer import analyze_model
from model_visualizer.interactive import InteractiveVisualizer
from model_visualizer.runtime import proveri_ulaze_modela
from model_visualizer.session import ComponentSpec, CompositeVisualizer, OverviewSpec

def visualize(model, example_inputs=None, example_kwargs=None, *, component_specs=None, overview_spec=None, validate_inputs=True, expanded_modules=None, show=True):
    """
    Glavni javni API VizTool-a.

    Standardni model:
        visualize(model)

    Model kome treba concrete input:
        visualize(
            model,
            example_inputs=x
        )

    Composite model:
        visualize(
            model,
            component_specs={...},
            overview_spec=OverviewSpec(...)
        )

    Ako je component_specs ili overview_spec prosleđen,
    koristi se jedna composite sesija.
    """

    composite_mode = component_specs is not None or overview_spec is not None

    if composite_mode:
        visualizer = CompositeVisualizer(model, component_specs=component_specs, overview_spec=overview_spec, validate_inputs=validate_inputs)
        if show:
            visualizer.show()
        return visualizer

    has_example_inputs = example_inputs is not None or bool(example_kwargs)

    if validate_inputs and has_example_inputs:
        proveri_ulaze_modela(model, example_inputs, example_kwargs)

    graph = analyze_model(model, example_inputs=example_inputs, example_kwargs=example_kwargs)
    visualizer = InteractiveVisualizer(graph, expanded_modules=expanded_modules)

    if show:
        visualizer.show()

    return visualizer

def visualize_components(model, component_specs=None, *, overview_spec=None, validate_inputs=True, show=True):
    """Kraći poziv za composite režim."""

    return visualize(
        model,
        component_specs=component_specs,
        overview_spec=overview_spec,
        validate_inputs=validate_inputs,
        show=show,
    )
