{{/*
Expand the name of the chart.
*/}}
{{- define "dnids.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "hdfs.name" -}}
{{- .Values.hdfs.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "spark.name" -}}
{{- .Values.spark.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "kafka.name" -}}
{{- .Values.kafka.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "dnids.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "dnids.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "dnids.labels" -}}
helm.sh/chart: {{ include "dnids.chart" . }}
{{ include "dnids.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "dnids.selectorLabels" -}}
app.kubernetes.io/name: {{ include "dnids.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
HDFS Image
*/}}
{{- define "hdfs.image" -}}
"{{ .Values.hdfs.image.repository }}:{{ .Values.hdfs.image.tag }}"
{{- end }}

{{/*
HDFS labels
*/}}
{{- define "hdfs.labels" -}}
helm.sh/chart: {{ include "dnids.chart" . }}
{{ include "hdfs.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
HDFS Selector labels
*/}}
{{- define "hdfs.selectorLabels" -}}
app.kubernetes.io/name: {{ include "hdfs.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Spark Image
*/}}
{{- define "spark.image" -}}
"{{ .Values.spark.image.repository }}:{{ .Values.spark.image.tag }}"
{{- end }}

{{/*
Spark Labels
*/}}
{{- define "spark.labels" -}}
helm.sh/chart: {{ include "dnids.chart" . }}
{{ include "spark.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Spark Selector labels
*/}}
{{- define "spark.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spark.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Kafka Image
*/}}
{{- define "kafka.image" -}}
"{{ .Values.kafka.image.repository }}:{{ .Values.kafka.image.tag }}"
{{- end }}

{{/*
Kafka Labels
*/}}
{{- define "kafka.labels" -}}
helm.sh/chart: {{ include "dnids.chart" . }}
{{ include "kafka.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Kafka Selector labels
*/}}
{{- define "kafka.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kafka.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}


{{/*
Create the name of the service account to use
*/}}
{{- define "dnids.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "dnids.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
