"""
OpenTelemetry tracing utilities for smolagents monitoring.
"""

import os
from typing import Optional
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


# Global variable to track if tracing has been initialized
_tracing_initialized = False


def setup_smolagents_tracing(
    endpoint: Optional[str] = None,
    enable_tracing: bool = True,
    resource_name: Optional[str] = None
) -> bool:
    """
    Set up OpenTelemetry tracing for smolagents.
    
    Args:
        endpoint: OTLP endpoint URL. Defaults to "http://0.0.0.0:6006/v1/traces"
        enable_tracing: Whether to enable tracing. Can be disabled for testing.
        resource_name: Custom resource name for the service. Defaults to "smolagents-service"
    
    Returns:
        bool: True if tracing was successfully initialized, False otherwise
    """
    global _tracing_initialized
    
    if not enable_tracing:
        return False
        
    if _tracing_initialized:
        return True
    
    try:
        # Use provided endpoint or default
        if endpoint is None:
            endpoint = os.getenv("OTLP_ENDPOINT", "http://0.0.0.0:6006/v1/traces")
        
        # Use provided resource name or default
        if resource_name is None:
            resource_name = "smolagents-service"
        
        # Create resource with project name for Arize Phoenix
        # Use the standard OpenTelemetry service.name attribute
        # Phoenix will use this to create separate projects
        resource = Resource.create({
            "service.name": resource_name,
            "endpoint": endpoint,
            "auto_instrumentation": True
        })
        
        # Set up trace provider with resource
        trace_provider = TracerProvider(resource=resource)
        
        # Create OTLP exporter with headers for project identification
        headers = {
            "x-project-name": resource_name,
            "x-service-name": resource_name,
            "x-endpoint": endpoint,
            "x-auto-instrumentation": True
        }
        
        trace_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(
                endpoint=endpoint,
                headers=headers
            ))
        )
        
        # Instrument smolagents
        SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
        
        _tracing_initialized = True
        return True
        
    except Exception as e:
        print(f"Warning: Failed to initialize tracing: {e}")
        return False


def is_tracing_enabled() -> bool:
    """Check if tracing has been initialized."""
    return _tracing_initialized


def reset_tracing():
    """Reset tracing state (useful for testing)."""
    global _tracing_initialized
    _tracing_initialized = False
