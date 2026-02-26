{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:

{% block methods %}
{% if methods %}
.. rubric:: Methods

.. autosummary::
   :toctree:
   :template: method.rst
{% for item in methods %}
{%- if not item.startswith('_') %}
   ~{{ name }}.{{ item }}
{%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes %}
{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :toctree:
{% for item in attributes %}
{%- if not item.startswith('_') %}
   ~{{ name }}.{{ item }}
{%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}