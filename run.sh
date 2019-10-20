
sudo docker run -it --runtime=nvidia --rm -e DISPLAY=$DISPLAY -v /home/michael/git/WikiArt_Feature_Maps:/src -v /home/michael/datasets/wikiart:/src/wikiart -w /src ufoym/deepo bash
