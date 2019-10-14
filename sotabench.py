from sotabencheval.image_classification import ImageNetEvaluator

evaluator = ImageNetEvaluator(
             model_name='ResNeXt-101-32x8d',
             paper_arxiv_id='1906.06423')

evaluator.add(dict(zip(image_ids, batch_output)))

evaluator.save()
