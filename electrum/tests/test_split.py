import random

from collections import defaultdict
from unittest import TestCase


def unique_hierarchy(hierarchy):
    new_hierarchy = defaultdict(list)
    for level, configs in hierarchy.items():
        unique_configs = set()
        for config in configs:
            # config dict can be out of order
            unique_configs.add(tuple((c, config[c]) for c in sorted(config.keys())))
        for unique_config in unique_configs:
            new_hierarchy[level].append(
                {t[0]: t[1] for t in unique_config})
    return new_hierarchy


def number_nonzero_shards(configuration):
    return len([v for v in configuration.values() if v])


class TestSplit(TestCase):
    def setUp(self) -> None:
        self.channels = {
            0: 1_000,
            1: 500,
            2: 302,
            3: 101,
        }

    def test_split(self):
        random.seed(1)
        self.suggest_splits(1100)

    def suggest_splits(self, amount_msat, attempts=0):
        MIN_SPLIT_AMOUNT_MSAT = 10

        # determine spendable amounts on channels
        channels_local_balances = {ck: cv for ck, cv in self.channels.items()}
        # 0. preselect channels that can send
        channels_with_funds = {ck: cv for ck, cv in
                               channels_local_balances.items() if
                               cv > MIN_SPLIT_AMOUNT_MSAT}

        def create_starting_split_hierarchy(amount_msat, channels_with_funds,
                                            configs=30):
            """Creates distribution of funds with least number of channels."""
            split_hierarchy = defaultdict(list)
            # shuffle to have different starting points
            for _ in range(configs):
                channels_order = list(channels_with_funds.keys())
                random.shuffle(channels_order)

                configuration = {}
                amount_added = 0
                for c in channels_order:
                    s = channels_with_funds[c]
                    if amount_added == amount_msat:
                        configuration[c] = 0
                    else:
                        amount_to_add = amount_msat - amount_added
                        amt = min(s, amount_to_add)
                        configuration[c] = amt
                        amount_added += amt
                if amount_added != amount_msat:
                    raise ValueError(
                        "Channels don't have enough sending capacity.")
                number_nonzero_elements = len(
                    [c for c in configuration.values() if c != 0])
                split_hierarchy[number_nonzero_elements].append(configuration)
            return unique_hierarchy(split_hierarchy)

        def propose_new_configuration(channels_with_funds, configuration,
                                      amount_msat, preserve_number_shards=True):
            # there are four basic operations to reach all states:
            def redistribute(config):
                # we redistribute the amount of already selected channels
                redistribution_amount = amount_msat // 10
                nonzero = [ck for ck, cv in config.items() if
                           cv >= redistribution_amount]
                # zero = [ck for ck, cv in configuration.items() if cv == 0]
                if len(
                        nonzero) == 1:  # we only have a single channel, so we can't redistribute
                    return config

                channel_from = random.choice(nonzero)
                channel_to = random.choice(nonzero)
                if channel_from == channel_to:
                    return config
                proposed_balance_from = config[
                                            channel_from] - redistribution_amount
                proposed_balance_to = config[channel_to] + redistribution_amount
                if (
                        proposed_balance_from < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to > channels_with_funds[channel_to] or
                        proposed_balance_from > channels_with_funds[channel_from]
                ):
                    return config
                else:
                    config[channel_from] = proposed_balance_from
                    config[channel_to] = proposed_balance_to
                assert sum([cv for cv in config.values()]) == amount_msat
                return config

            def split(config):
                # we split the amount sent from a channel to another channel
                nonzero = [ck for ck, cv in config.items() if cv != 0]
                zero = [ck for ck, cv in config.items() if cv == 0]
                try:
                    channel_from = random.choice(nonzero)
                    channel_to = random.choice(zero)
                except IndexError:
                    return config
                delta = config[channel_from] // 10
                proposed_balance_from = config[channel_from] - delta
                proposed_balance_to = config[channel_to] + delta
                if (
                        proposed_balance_from < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to > channels_with_funds[channel_to] or
                        proposed_balance_from > channels_with_funds[channel_from]
                ):
                    return config
                else:
                    config[channel_from] = proposed_balance_from
                    config[channel_to] = proposed_balance_to
                    assert sum([cv for cv in config.values()]) == amount_msat
                return config

            def swap(config):
                # we swap the amounts from a single channel with another channel
                nonzero = [ck for ck, cv in config.items() if cv != 0]
                all = list(config.keys())

                channel_from = random.choice(nonzero)
                channel_to = random.choice(all)

                proposed_balance_to = config[channel_from]
                proposed_balance_from = config[channel_to]
                if (
                        proposed_balance_from < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to < MIN_SPLIT_AMOUNT_MSAT or
                        proposed_balance_to > channels_with_funds[channel_to] or
                        proposed_balance_from > channels_with_funds[channel_from]
                ):
                    return config
                else:
                    config[channel_to] = proposed_balance_to
                    config[channel_from] = proposed_balance_from
                return config

            initial_number_shards = number_nonzero_shards(configuration)

            for _ in range(5):
                configuration = redistribute(configuration)
            if not preserve_number_shards and number_nonzero_shards(
                    configuration) == initial_number_shards:
                configuration = split(configuration)
            configuration = swap(configuration)

            return configuration

        # create starting split configurations with different split levels
        split_hierarchy = create_starting_split_hierarchy(amount_msat, channels_with_funds)
        print(split_hierarchy)

        CANDIDATES_PER_LEVEL = 20
        # generate splittings of different split levels
        for level in range(2, min(5, len(self.channels) + 1)):  # start with two splittings
            for _ in range(
                    CANDIDATES_PER_LEVEL):  # generate a set of configurations for each level
                configurations = unique_hierarchy(split_hierarchy).get(level, None)
                if configurations:  # we have a splitting of the desired number of shards
                    configuration = random.choice(configurations)
                    # generate new splittings preserving the number of shards
                    configuration = propose_new_configuration(
                        channels_with_funds, configuration, amount_msat,
                        preserve_number_shards=True)
                else:
                    # go one level deeper and look for valid splttings, try to generate
                    # from there
                    configurations = unique_hierarchy(split_hierarchy).get(level - 1, None)
                    if not configurations:
                        raise ValueError(
                            "No more configurations can be found")
                    configuration = random.choice(configurations)
                    # generate new splittings going one level higher in the number of shards
                    configuration = propose_new_configuration(
                        channels_with_funds, configuration, amount_msat,
                        preserve_number_shards=False)

                # add the newly found configuration (doesn't matter if nothing changed)
                split_hierarchy[
                    number_nonzero_shards(configuration)].append(
                    configuration)

        hierarchy = unique_hierarchy(split_hierarchy)
        for level, configs in hierarchy.items():
            # print(level, configs)
            for config in configs:
                assert number_nonzero_shards(config) == level, f"{config} {level}"

        def rate_configuration(config, shard_penalty):
            F = 0
            amount = sum([v for v in config.values()])

            for channel, value in config.items():
                if value:
                    value /= amount  # normalize
                    F += value * value + shard_penalty * shard_penalty
            return F

        def rate_configurations(hierarchy, shard_penalty):
            rated_configs = []
            for level, configs in hierarchy.items():
                for config in configs:
                    rated_configs.append((config, rate_configuration(config, shard_penalty)))
            return rated_configs

        print("Channels with sendable amounts:")
        print(channels_local_balances)
        print()
        shard_penalty = 1.00
        print(f"Final rated configurations for sending a value of {amount_msat}, shard penalty {shard_penalty}, min shard {MIN_SPLIT_AMOUNT_MSAT}:")
        for config_rating in sorted(rate_configurations(hierarchy, shard_penalty=shard_penalty), key=lambda c: c[1], reverse=False):
            print(number_nonzero_shards(config_rating[0]), config_rating[0], round(config_rating[1], 3))