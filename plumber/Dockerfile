FROM rstudio/plumber
RUN apt-get update -qq && apt-get install -y libssl-dev libcurl4-gnutls-dev libpng-dev libpng-dev pandoc
RUN R -e "install.packages(c('tidymodels', 'forcats', 'dplyr', 'readr', 'tidyverse', 'ggplot2', 'yardstick', 'GGally', 'leaflet', 'ranger', 'plumber'))"
COPY diabetes_binary_health_indicators_BRFSS2015.csv diabetes_binary_health_indicators_BRFSS2015.csv .
COPY API.R API.R
EXPOSE 8000
ENTRYPOINT ["R", "-e", \
"pr <- plumber::plumb('API.R'); pr$run(host='0.0.0.0', port=8000)"]
