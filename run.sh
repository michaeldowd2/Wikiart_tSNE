
sudo docker run -it --runtime=nvidia --rm -e DISPLAY=$DISPLAY -v /home/michael/git/wikiart_feature_maps:/src -v /home/michael/datasets/wikiart:/src/wikiart -w /src ufoym/deepo bash
