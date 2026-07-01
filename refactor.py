import os

index_path = r'c:\private\Project\SmartCCTV\dashboard\templates\dashboard\index.html'
base_path = r'c:\private\Project\SmartCCTV\dashboard\templates\dashboard\base.html'
live_path = r'c:\private\Project\SmartCCTV\dashboard\templates\dashboard\live_monitoring.html'

with open(index_path, 'r', encoding='utf-8') as f:
    index_content = f.read()

# Find split points
head_end = index_content.find('</head>')
sidebar_start = index_content.find('<nav class="sidebar" id="sidebar">')
main_content_start = index_content.find('<div class="content-inner">') + len('<div class="content-inner">')
main_content_end = index_content.find('</div><!-- /content-inner -->')
js_scripts_start = index_content.find('<!-- Bootstrap JS -->', main_content_end)

if js_scripts_start == -1:
    js_scripts_start = index_content.find('<script src="https://cdn.jsdelivr.net/npm/bootstrap', main_content_end)

header_part = index_content[:head_end] + '    {% block extra_css %}{% endblock %}\n' + index_content[head_end:main_content_start]
footer_part = '\n                {% block content %}{% endblock %}\n            ' + index_content[main_content_end:js_scripts_start]
scripts_part = index_content[js_scripts_start:index_content.find('<!-- ── CHART INITIALISATIONS')]

base_content = header_part + footer_part + scripts_part + '{% block extra_js %}{% endblock %}\n</body>\n</html>\n'

# Update sidebar active states
base_content = base_content.replace(
    '<a class="nav-link-item active" href="{% url \'home\' %}">',
    '<a class="nav-link-item {% if request.resolver_match.url_name == \'home\' %}active{% endif %}" href="{% url \'home\' %}">'
)
base_content = base_content.replace(
    '<a class="nav-link-item" href="{% url \'live_monitoring\' %}">',
    '<a class="nav-link-item {% if request.resolver_match.url_name == \'live_monitoring\' %}active{% endif %}" href="{% url \'live_monitoring\' %}">'
)
base_content = base_content.replace('<title>Smart CCTV Monitoring System</title>', '<title>{% block title %}Smart CCTV Monitoring System{% endblock %}</title>')

with open(base_path, 'w', encoding='utf-8') as f:
    f.write(base_content)

# Update index.html
new_index_content = '{% extends "dashboard/base.html" %}\n{% load static %}\n{% block content %}\n'
new_index_content += index_content[main_content_start:main_content_end]
new_index_content += '\n{% endblock %}\n\n{% block extra_js %}\n'
new_index_content += index_content[index_content.find('<!-- ── CHART INITIALISATIONS'):]
new_index_content = new_index_content.replace('</body>\n</html>', '').replace('</div><!-- /page-wrapper -->', '')
new_index_content += '{% endblock %}\n'

with open(index_path, 'w', encoding='utf-8') as f:
    f.write(new_index_content)

# Process live_monitoring.html
with open(live_path, 'r', encoding='utf-8') as f:
    live_content = f.read()

live_style_start = live_content.find('<style>')
live_style_end = live_content.find('</style>') + 8
live_css = live_content[live_style_start:live_style_end]

live_body_start = live_content.find('<!-- HEADER -->')
live_body_end = live_content.find('<!-- Bootstrap JS -->')

live_scripts_start = live_content.find('<script>', live_body_end)
live_scripts_end = live_content.rfind('</body>')

new_live_content = '{% extends "dashboard/base.html" %}\n{% load static %}\n\n{% block title %}Live Monitoring - Smart CCTV{% endblock %}\n\n{% block extra_css %}\n'
new_live_content += live_css + '\n{% endblock %}\n\n{% block content %}\n'
new_live_content += live_content[live_body_start:live_body_end]
new_live_content += '\n{% endblock %}\n\n{% block extra_js %}\n'
new_live_content += live_content[live_scripts_start:live_scripts_end]
new_live_content += '\n{% endblock %}\n'

with open(live_path, 'w', encoding='utf-8') as f:
    f.write(new_live_content)

print("Template refactoring completed successfully.")
