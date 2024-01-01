# Use an official Ubuntu runtime as a base image
FROM ubuntu:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update the system and install necessary packages
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        wget
 

# Install Python libraries
RUN pip install pandas numpy transformers  numpy torch torchvision 
RUN pip install scikit-learn BeautifulSoup4

# Set the working directory
WORKDIR /project
COPY /directory/ /project/
COPY script.py /project/

# Start a shell by default
ENTRYPOINT ["python3", "script.py"]